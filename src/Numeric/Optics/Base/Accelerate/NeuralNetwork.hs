{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Numeric.Optics.Base.Accelerate.NeuralNetwork where

import Numeric.Optics
import Numeric.Optics.Base.Accelerate as Base

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Numeric.LinearAlgebra ((#>), (<.>), Numeric)
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as LA

repeated :: Elt t => Exp Int -> Acc (Vector t) -> Acc (Matrix t)
repeated n = A.replicate (I2 n $ constant All)

-- n x na matrix full of copies of v, with a = |v|
blockVector :: Elt t => Exp Int -> Acc (Vector t) -> Acc (Matrix t)
blockVector b v = reshape (I2 b (b * a)) $ repeated (b * b) v
  where a = size v :: Exp Int

-- | The jacobian of a linear layer with @M : a → b@, @v : a@
-- is a linear map @J_L : ba + a → b@ represented by the following block matrix:
--
-- >>> :{
--    | v ...(b) ... v    |   |
--    | ⋮(b)         ⋮(b) | M |
--    | v ...(b) ... v    |   |
-- :}
linearLayerJacobian :: Elt t => Acc (Matrix t) -> Acc (Vector t) -> Acc (Matrix t)
linearLayerJacobian m v = blockVector b v A.++ m
  where (I2 b _) = shape m

-- TODO: assert that vector, matrix dimensions match, i.e. that
--       @m : a → b@, @x : a@, and @y : b@
-- TODO: use the more efficient way to compute this that exploits the block
-- structure of @J@, which is something like
--
-- >>> :{
--    sum (v_1 y) = v_1 (Σ y)
--    sum (v_2 y)
--    ⋮
--    M y
-- :}
--
-- So the first part is b (?) repetitions of (Σy) v
--
-- TODO: tests: y = 0 => (m', x') = 0
linearLayerReverseDerivative :: (Elt t, Numeric t)
  => Acc (Matrix t) -- ^ A @(b × a)@ matrix @m@  representing a linear map @m : (a → b)@
  -> Acc (Vector t) -- ^ A vector @x : a@
  -> Acc (Vector t) -- ^ A vector of *changes* @y : b'@
  -> Acc (Matrix t, Vector t) -- ^ resulting changes in @m : a → b@ and @x : a@
linearLayerReverseDerivative m x y = T2 m' x' where
  (I2 b a) = shape m
  r = transpose (linearLayerJacobian m x) #> y
  m' = reshape (I2 b a) $ A.take (b * a) r
  x' = A.drop (b * a) r

-- | NOTE: the *dimensions* of this are not type-safe; k is just the underlying
-- field of the matrix. We would ideally write
-- @A.Matrix k a b -> A.Vector k a -> A.Vector k b@
linearLayer :: forall k . Numeric k => MonoLens (,) (DSL Acc) (Matrix k, Vector k) (Vector k)
linearLayer = MonoLens (DSL f) (DSL f') where
  f :: Acc (Matrix k, Vector k) -> Acc (Vector k)
  f (T2 m x) = m #> x

  f' :: Acc ((Matrix k, Vector k), Vector k) -> Acc (Matrix k, Vector k)
  f' (T2 (T2 m x) y) = linearLayerReverseDerivative m x y

activationLayer :: forall k . Numeric k
  => MonoLens (,) (DSL Exp) k k -- ^ An activation function and its reverse derivative
  -> MonoLens (,) (DSL Acc) (Vector k) (Vector k)
activationLayer = Base.mapLens

-- | A 'dense' neural network layer multiplies a dense matrix of weights by a vector
-- of inputs, and applies an activation function
dense :: forall k . Numeric k
  => MonoLens (,) (DSL Exp) k k -- ^ activation function lens e.g., 'sigmoid'
  -> MonoLens (,) (DSL Acc) (Matrix k, Vector k) (Vector k)
dense activation = linearLayer ~> mapLens activation

-------------------------------
-- activation functions & simple numeric lenses on Exps

-- TODO: double check implementation
sigmoid :: forall k . (A.Num k, A.Floating k) => MonoLens (,) (DSL Exp) k k
sigmoid = MonoLens (DSL f) (DSL f') where
  f x = 1 / (1 + exp (negate x))
  f' (T2 x y) = f x * (1 - f x) * y
