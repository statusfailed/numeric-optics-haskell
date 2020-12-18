{-# LANGUAGE PatternSynonyms #-}
-- ^ Needed to import Accelerate's T2 pattern synonym
{-# LANGUAGE TypeFamilies #-}
-- ^ Needed for Distributes instance
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
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
    h' = ex ~> (f' × g')
    -- σ : (A × B) → (B × A)
    σ  = pair π1 π0
    -- ex : (A × C) × (B' × D') → (A × B') × (C × D')
    ex = assocL ~> (id × (assocR ~> (σ × id) ~> assocL)) ~> assocR

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
    h' = ex ~> (f' × g')
    -- σ : (A × B) → (B × A)
    σ  = pair π1 π0
    -- ex : (A × C) × (B' × D') → (A × B') × (C × D')
    ex = assocL ~> (id × (assocR ~> (σ × id) ~> assocL)) ~> assocR

  -- TODO: NOTE: ask everyone if these are right!
  assocL = MonoLens assocL (proj1 ~> assocR)
  assocR = MonoLens assocR (proj1 ~> assocL)

  unitorL = MonoLens unitorL (proj1 ~> counitorL)
  counitorL = MonoLens counitorL (proj1 ~> unitorL)

-------------------------------
-- Combinators

-- | Map an 'Exp' *lens* over an array.
mapLens :: (Elt a, Elt b, Shape sh)
  => MonoLens (,) (DSL Exp) a b
  -> MonoLens (,) (DSL Acc) (Array sh a) (Array sh b)
mapLens (MonoLens f f') = MonoLens h h' where
  h  = translate A.map f
  h' = translate (A.uncurry . A.zipWith . A.curry) f'
