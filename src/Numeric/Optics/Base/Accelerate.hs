{-# LANGUAGE PatternSynonyms #-}
-- ^ Needed to import Accelerate's T2 pattern synonym
{-# LANGUAGE TypeFamilies #-}
-- ^ Needed for Distributes instance
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
module Numeric.Optics.Base.Accelerate where

import Prelude hiding (id)

import Control.Categories
import Numeric.Optics.Types

import qualified Data.Array.Accelerate as A
import Data.Array.Accelerate (Acc, Exp, Arrays, Elt, pattern T2, Array, Shape)

-------------------------------
-- Base category (1): Accelerate array programs

instance Cat (DSL Acc) where
  type Obj (DSL Acc) a = Arrays a
  id = DSL id
  (DSL f) ~> (DSL g) = DSL (f ~> g)

instance Monoidal (DSL Acc) where
  type Tensor (DSL Acc) a b = (a, b)
  type Unit   (DSL Acc)     = ()

  tensor (DSL f) (DSL g) = DSL $ \(T2 a c) -> T2 (f a) (g c)

  unitorL = DSL A.asnd
  counitorL = DSL (\a -> A.T2 (A.lift ()) a)

  assocL  = DSL $ \(T2 (T2 a b) c) -> T2 a (T2 b c)
  assocR  = DSL $ \(T2 a (T2 b c)) -> T2 (T2 a b) c

instance Cartesian (DSL Acc) where
  proj0 = DSL A.afst
  proj1 = DSL A.asnd
  pair (DSL f) (DSL g) = DSL (\q -> T2 (f q) (g q))

-------------------------------
-- Base category (2): Accelerate Expressions
-- NOTE: these instances essentially the same as for DSL Acc.
-- Can we remove boilerplate using the 'Distributes' class?

instance Cat (DSL Exp) where
  type Obj (DSL Exp) a = Elt a
  id = DSL id
  (DSL f) ~> (DSL g) = DSL (f ~> g)

instance Monoidal (DSL Exp) where
  type Tensor (DSL Exp) a b = (a, b)
  type Unit   (DSL Exp)     = ()

  tensor (DSL f) (DSL g) = DSL $ \(T2 a c) -> T2 (f a) (g c)

  unitorL = DSL A.snd
  counitorL = DSL (\a -> A.T2 (A.lift ()) a)

  assocL  = DSL $ \(T2 (T2 a b) c) -> T2 a (T2 b c)
  assocR  = DSL $ \(T2 a (T2 b c)) -> T2 (T2 a b) c

instance Cartesian (DSL Exp) where
  proj0 = DSL A.fst
  proj1 = DSL A.snd
  pair (DSL f) (DSL g) = DSL (\q -> T2 (f q) (g q))

-------------------------------
-- MonoLens instances for DSL Acc

instance Cat (MonoLens (,) (DSL Acc)) where
  type Obj (MonoLens (,) (DSL Acc)) a = Obj (DSL Acc) a
  id = MonoLens id proj1
  (MonoLens f f') ~> (MonoLens g g') = MonoLens h h' where
    h  = (f ~> g)
    h' = (pair id f `tensor` id) ~> assocL ~> (id `tensor` g') ~> f'

instance Monoidal (MonoLens (,) (DSL Acc)) where
  type Tensor (MonoLens (,) (DSL Acc)) a b = (a, b)
  type Unit   (MonoLens (,) (DSL Acc))     = ()

  tensor (MonoLens f f') (MonoLens g g') = MonoLens h h' where
    h  = f × g
    h' = exchange ~> (f' × g')

  -- TODO: NOTE: ask everyone if these are right!
  assocL = MonoLens assocL (proj1 ~> assocR)
  assocR = MonoLens assocR (proj1 ~> assocL)

  unitorL = MonoLens unitorL (proj1 ~> counitorL)
  counitorL = MonoLens counitorL (proj1 ~> unitorL)

-------------------------------
-- MonoLens instances for DSL Exp
-- NOTE: these instances genuinely are just copy-pased with types changed;
-- I haven't been able to write a proper generic instance for
-- @Cartesian cat => MonoLens (Tensor cat) cat@.

instance Cat (MonoLens (,) (DSL Exp)) where
  type Obj (MonoLens (,) (DSL Exp)) a = Obj (DSL Exp) a
  id = MonoLens id proj1
  (MonoLens f f') ~> (MonoLens g g') = MonoLens h h' where
    h  = (f ~> g)
    h' = (pair id f `tensor` id) ~> assocL ~> (id `tensor` g') ~> f'

instance Monoidal (MonoLens (,) (DSL Exp)) where
  type Tensor (MonoLens (,) (DSL Exp)) a b = (a, b)
  type Unit   (MonoLens (,) (DSL Exp))     = ()

  tensor (MonoLens f f') (MonoLens g g') = MonoLens h h' where
    h  = f × g
    h' = exchange ~> (f' × g')

  -- TODO: NOTE: ask everyone if these are right!
  assocL = MonoLens assocL (proj1 ~> assocR)
  assocR = MonoLens assocR (proj1 ~> assocL)

  unitorL = MonoLens unitorL (proj1 ~> counitorL)
  counitorL = MonoLens counitorL (proj1 ~> unitorL)

-------------------------------
-- Combinators and array-valued lenses
-- TODO: put these in a "Lenses" module so it can be imported qualified?

-- | Map an 'Exp' *lens* over an array.
mapLens :: (Elt a, Elt b, Shape sh)
  => MonoLens (,) (DSL Exp) a b
  -> MonoLens (,) (DSL Acc) (Array sh a) (Array sh b)
mapLens (MonoLens f f') = MonoLens h h' where
  h  = translate A.map f
  h' = translate (A.uncurry . A.zipWith . A.curry) f'

-- | The lens equivalent of zipWith
zipWithLens :: (Elt a, Elt b, Elt c, Shape sh)
  => MonoLens (,) (DSL Exp) (a, b) c
  -> MonoLens (,) (DSL Acc) (Array sh a, Array sh b) (Array sh c)
zipWithLens (MonoLens f f') = MonoLens h (DSL h') where
  h  = translate (A.uncurry . A.zipWith . A.curry) f

  -- NOTE: Running f' pointwise results in an array of (a, b), but what we
  -- really want is two arrays (Array a, Array b).
  -- We therefore have to map "fst" and "snd" over the array of pairs to get a
  -- pair of arrays.
  -- TODO: Check if there's a better way to do this!
  h' (T2 (T2 as bs) cs) =
    let r = A.zipWith3 k as bs cs
    in  T2 (A.map A.fst r) (A.map A.snd r)

  k a b c = runDSL f' (T2 (T2 a b) c)

-- Array constants
fillLens :: (Shape sh, Elt a)
  => Exp sh
  -> Exp a
  -> MonoLens (,) (DSL Acc) () (Array sh a)
fillLens sh a = MonoLens (DSL f) (DSL f') where
  f  _ = A.fill sh a
  f' _ = A.lift ()

-- NOTE: this is an example of an isomorphism and its reverse derivative;
-- the forward and reverse functions only change types.
reshapeLens :: (Shape sh, Shape sh', Elt a)
  => Exp sh
  -> MonoLens (,) (DSL Acc) (Array sh' a) (Array sh a)
reshapeLens sh = MonoLens (DSL f) (DSL f') where
  f           = A.reshape sh
  f' (T2 x y) = A.reshape (A.shape x) y

projection :: (A.Shape sh, A.Eq sh, A.Num a) => Exp sh -> MonoLens (,) (DSL Acc) (Array sh a) (A.Scalar a)
projection sh = MonoLens (DSL f) (DSL f') where
  f  x        = A.unit (x A.! sh)
  f' (T2 x y) = A.generate (A.shape x) (\sh' -> sh A.== sh' A.? (A.the y, 0))

-------------------------------
-- Scalar lenses

multiply :: A.Num a => ParaLens (,) (DSL Exp) a a a
multiply = MonoLens (DSL f) (DSL f') where
  f  (T2 p a)        = p A.* a
  f' (T2 (T2 p a) b) = T2 (a A.* b) (p A.* b)

add :: A.Num a => ParaLens (,) (DSL Exp) a a a
add = MonoLens (DSL f) (DSL f') where
  f  (T2 p a)        = p A.+ a
  f' (T2 (T2 p a) b) = T2 b b

-- NOTE: TODO: the reverse component of this lens is NOT the reverse derivative
-- of "floor"- this is basically a use of the "straight-through estimator".
efloor :: (A.Ord a, A.Fractional a, A.RealFrac a, A.Integral b, A.FromIntegral A.Int64 b, A.FromIntegral b a)
  => MonoLens (,) (DSL Exp) a b
efloor = MonoLens (DSL f) (DSL f') where
  f  = A.floor
  f' = A.fromIntegral . A.snd

eround :: (A.RealFrac a, A.FromIntegral b a, A.Elt a, A.Ord b, A.Integral b, A.FromIntegral A.Int64 b)
  => MonoLens (,) (DSL Exp) a b
eround = MonoLens (DSL f) (DSL f') where
  f  = A.round
  f' = A.fromIntegral . A.snd
