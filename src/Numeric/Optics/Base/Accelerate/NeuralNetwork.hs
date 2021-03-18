{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Numeric.Optics.Base.Accelerate.NeuralNetwork where

import Numeric.Optics
-- TODO: this imports a higher-level module which is kinda weird
import Numeric.Optics.Base.Accelerate as Base

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Numeric.LinearAlgebra ((#>), (><), Numeric)
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as LA

-- | NOTE: the *dimensions* of this are not type-safe; k is just the underlying
-- field of the matrix.
-- TODO: Use type-level naturals to enforce size-safety
-- (maybe type-safe Matrix/Vector types?)
linearLayer :: forall k . Numeric k => MonoLens (,) (DSL Acc) (Matrix k, Vector k) (Vector k)
linearLayer = MonoLens (DSL f) (DSL f') where
  f :: Acc (Matrix k, Vector k) -> Acc (Vector k)
  f (T2 m x) = m #> x

  -- See the following link for a derivation:
  -- https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
  f' :: Acc ((Matrix k, Vector k), Vector k) -> Acc (Matrix k, Vector k)
  f' (T2 (T2 m x) y) = T2 (y >< x) (A.transpose m #> y)

-- | An activation layer for activation function @f@ is a scalar lens applied
-- pointwise to each element of an array, i.e,. @mapLens f@.
activationLayer :: forall k . Numeric k
  => MonoLens (,) (DSL Exp) k k -- ^ An activation function and its reverse derivative
  -> MonoLens (,) (DSL Acc) (Vector k) (Vector k)
activationLayer = Base.mapLens

-- | A 'dense' neural network layer multiplies a dense matrix of weights by a vector
-- of inputs, and applies an activation function
dense :: forall k . Numeric k
  => MonoLens (,) (DSL Exp) k k -- ^ activation function lens e.g., 'sigmoid'
  -> ParaLens (,) (DSL Acc) (Vector k, Matrix k) (Vector k) (Vector k)
dense activation = linearLayer ~~> zipWithLens add ~> mapLens activation

-------------------------------
-- activation functions & simple numeric lenses on Exps

-- TODO: double check implementations of sigmoid, tanh.

sigmoid :: forall k . (A.Num k, A.Floating k) => MonoLens (,) (DSL Exp) k k
sigmoid = MonoLens (DSL f) (DSL f') where
  f x = 1 / (1 + exp (negate x))
  f' (T2 x y) = f x * (1 - f x) * y

tanh :: forall k . (A.Num k, A.Floating k) => MonoLens (,) (DSL Exp) k k
tanh = MonoLens (DSL f) (DSL f') where
  f = A.tanh
  f' (T2 x y) = (1 A.- f x A.* f x) A.* y
