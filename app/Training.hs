{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- Below here is stuff used in 'step'
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
module Training where

import Prelude hiding (id, (<))

import Numeric.Optics
import Numeric.Optics.Base.Accelerate
import Numeric.Optics.Base.Accelerate.NeuralNetwork as NN

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Numeric.LinearAlgebra

-------------------------------
-- Classes to allow defining update lenses
-- TODO: move these somewhere more appropriate, and maybe generalise them a bit.

class Scale s v where
  scale :: s -> v -> v

instance A.Num s => Scale (Exp s) (Acc ()) where
  scale :: Exp s -> Acc () -> Acc ()
  scale s v = v

instance A.Num s => Scale (Exp s) (Acc (Vector s)) where
  scale :: Exp s -> Acc (Vector s) -> Acc (Vector s)
  scale s v = A.map (* s) v

instance A.Num s => Scale (Exp s) (Acc (Matrix s)) where
  scale :: Exp s -> Acc (Matrix s) -> Acc (Matrix s)
  scale s v = A.map (* s) v

-- TODO: remove this instance?
instance A.Num s => Scale (Acc (Scalar s)) (Acc (Vector s)) where
  scale :: Acc (Scalar s) -> Acc (Vector s) -> Acc (Vector s)
  scale s v = A.map (* the s) v

instance A.Num s => Scale (Acc (Scalar s)) (Acc (Matrix s)) where
  scale :: Acc (Scalar s) -> Acc (Matrix s) -> Acc (Matrix s)
  scale s v = A.map (* the s) v

instance (Scale s v, Scale s w) => Scale s (v, w) where
  scale s (v, w) = (scale s v, scale s w)

instance (Arrays v, Arrays w, Scale s (Acc v), Scale s (Acc w)) => Scale s (Acc (v, w)) where
  scale s (T2 v w) = T2 (scale s v) (scale s w)

class PointwiseAdditive v where
  pointwiseAdd :: v -> v -> v
  pointwiseNegate :: v -> v

instance PointwiseAdditive (Acc ()) where
  pointwiseAdd x _ = x
  pointwiseNegate = id

instance (A.Num a, Shape sh) => PointwiseAdditive (Acc (Array sh a)) where
  pointwiseAdd :: Acc (Array sh a) -> Acc (Array sh a) -> Acc (Array sh a)
  pointwiseAdd v w = A.zipWith (A.+) v w
  pointwiseNegate = A.map A.negate

instance (PointwiseAdditive v, PointwiseAdditive w) => PointwiseAdditive (v, w) where
  pointwiseAdd (v, w) (v', w') = (pointwiseAdd v v', pointwiseAdd w w')
  pointwiseNegate (v, w) = (pointwiseNegate v, pointwiseNegate w)

instance (Arrays v, Arrays w, PointwiseAdditive (Acc v), PointwiseAdditive (Acc w)) => PointwiseAdditive (Acc (v, w)) where
  pointwiseAdd (T2 v w) (T2 v' w') = T2 (pointwiseAdd v v') (pointwiseAdd w w')
  pointwiseNegate (T2 v w) = T2 (pointwiseNegate v) (pointwiseNegate w)

-------------------------------
-- A standard gradient descent update lens

-- Vanilla gradient descent update lens
-- TODO: Add Num constraint for ε so we can negate it before scaling?
gdUpdate :: (Arrays p, Scale ε (Acc p), PointwiseAdditive (Acc p))
  => ε -- ^ learning rate
  -> MonoLens (,) (DSL Acc) p p
gdUpdate ε = MonoLens id (DSL f') where
  f' (T2 p p') = pointwiseAdd p (pointwiseNegate $ scale ε p')

meanSquaredErrorDisplacement :: (Arrays b, PointwiseAdditive (Acc b)) => MonoLens (,) (DSL Acc) b b
meanSquaredErrorDisplacement = MonoLens f (DSL f') where
  f  = id
  f' (T2 yhat ytrue) = pointwiseAdd yhat (pointwiseNegate ytrue)

cceDisplacement :: (A.Ord b, A.Fractional b, Shape sh) => MonoLens (,) (DSL Acc) (Array sh b) (Array sh b)
cceDisplacement = MonoLens f (DSL f') where
  f  = id
  f' (T2 yhat ytrue) = pointwiseNegate $ A.zipWith (/) ytrue (A.map (clip ε (1 - ε)) yhat)

  -- use of ε with clip prevents divide-by-zero errors
  -- TODO pass ε as a parameter to cceDisplacement
  ε  = 0.01

-- TODO: We're missing the inverse of the displacement map
-- TODO: rename this
step :: forall t cat p a b . (Arrays p, Arrays a, Arrays b)
  => MonoLens (,) (DSL Acc) p p
  -> MonoLens (,) (DSL Acc) b b
  -> ParaLens (,) (DSL Acc) p a b
  -> MonoLens (,) (DSL Acc) (p, a) b -- ^ Note: running the *reverse* of this lens gets us updated parameters & data!
step update displacement model = tensor update id ~> model ~> displacement

-------------------------------
-- Evaluation

afoldrN :: Arrays b => Exp Int -> (Exp Int -> Acc a) -> (Acc a -> Acc b -> Acc b) -> Acc b -> Acc b
afoldrN n load f b0 = A.asnd $ A.awhile condition g (T2 (unit 0) b0) where
  condition (T2 i _) = unit (the i A.< n)
  g (T2 i b) = T2 (unit $ the i A.+ 1) (f (load (the i)) b)

emptyVector :: Elt a => Vector a
emptyVector = A.fromList (Z :. 0) []

-- | Apply a vector function to each index of a matrix, but don't reshape the
-- output.
vmap' :: (Elt a, Elt b) => (Acc (Vector a) -> Acc (Vector b)) -> Acc (Matrix a) -> Acc (Vector b)
vmap' f a = afoldrN n load g (use emptyVector) where
  (I2 n _) = shape a
  load i = A.slice a (lift $ Z :. i :. All)
  g a b = b A.++ f a

-- | Apply a vector function to each index of a matrix.
-- NOTE: @vmap f m@ is only safe if @f a@ has the same length
-- for each @a@ in @m@
vmap :: (Elt a, Elt b) => (Acc (Vector a) -> Acc (Vector b)) -> Acc (Matrix a) -> Acc (Matrix b)
vmap f a = A.reshape (I2 n (nk `A.div` n)) b where
  b = vmap' f a
  (I2 n _) = shape a
  (I1 nk) = shape b

-- | Return a boolean vector of those instances (vectors) that were correctly
-- classified by a function.
correctlyClassified :: (Elt a, Elt b, A.Eq b)
  => (Acc (Vector a) -> Acc (Vector b)) -> Acc (Matrix a) -> Acc (Matrix b) -> Acc (Vector Bool)
correctlyClassified f a b = A.all id $ A.zipWith (A.==) b (vmap f a)

-- | Counts the number of @True@ values in a vector
accuracy :: Acc (Vector Bool) -> Exp Double
accuracy v = count v / len v where
  count = the . A.sum . A.map (? (1,0))
  len   = A.fromIntegral . A.length

-------------------------------
-- Utilities

toVector :: Elt a => Acc (Scalar a) -> Acc (Vector a)
toVector = A.reshape (I1 1)

enumAll :: (Prelude.Enum a, Prelude.Bounded a) => [a]
enumAll = enumFromTo minBound maxBound

clip :: A.Ord e => Exp e -> Exp e -> Exp e -> Exp e
clip lower upper = A.min upper . A.max lower
