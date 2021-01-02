{-# LANGUAGE KindSignatures #-}
-- ^ Lets us be explicit about kinds in the type of 'Distributes'
{-# LANGUAGE NoStarIsType   #-}
-- ^ We use Type instead of * in kind signatures
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-- ^ Needed to create instances of Cat, Monoidal, etc.
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
-- ^ needed to define parametrised composition (~~>)
module Numeric.Optics.Types where

import Prelude hiding (id)
import GHC.Types (Type)
import GHC.Exts (Constraint)
import Control.Categories

-- | Monomorphic lenses over a category and tensor product
data MonoLens tensor cat a b = MonoLens (cat a b) (cat (tensor a b) a)

-- Simple instance for MonoLens over Hask with (,) as tensor product
instance Cat (MonoLens (,) (->)) where
  type Obj (MonoLens (,) (->)) a = Obj (->) a
  id = MonoLens id proj1
  (MonoLens f f') ~> (MonoLens g g') = MonoLens h h' where
    h = f ~> g
    h' = (pair id f `tensor` id) ~> assocL ~> (id `tensor` g') ~> f'

instance Monoidal (MonoLens (,) (->)) where
  type Tensor (MonoLens (,) (->)) a b = (a, b)
  type Unit   (MonoLens (,) (->))     = ()

  tensor (MonoLens f f') (MonoLens g g') = MonoLens h h' where
    h  = f × g
    h' = exchange ~> (f' × g')

  -- TODO: NOTE: ask everyone if these are right!
  assocL = MonoLens assocL (proj1 ~> assocR)
  assocR = MonoLens assocR (proj1 ~> assocL)

  unitorL = MonoLens unitorL (proj1 ~> counitorL)
  counitorL = MonoLens counitorL (proj1 ~> unitorL)

-------------------------------
-- useful combinators

-- | The type of "context-wrapped" DSL functions.
-- We use this type to provide Category instances for Accelerate.
newtype DSL dsl a b = DSL { runDSL :: dsl a -> dsl b }

-- | Translate between two DSLs
-- For example using Data.Array.Accelerate,
-- >>> translate A.map
-- is a map which turns a scalar Accelerate function @f@ into an array
-- function, by mapping @f@ over each element.
translate :: ((dsl1 a1 -> dsl1 b1) -> (dsl2 a2 -> dsl2 b2)) -> DSL dsl1 a1 b1 -> DSL dsl2 a2 b2
translate f = DSL . f . runDSL

-------------------------------
-- | Parametrised morphisms
-- NOTE: Instead of defining Para as a category as in the paper, we instead
-- provide combinators for parametrised composition and tensor.
-- (this is only because I haven't been able to convince GHC that Para is a category)
type Para tensor cat p a b = cat (tensor p a) b

-- | Parametrised monomorphic lens morphisms
type ParaLens tensor cat p a b = MonoLens tensor cat (tensor p a) b

-- | Parametrised composition
-- >>> (~~>) : (P × A → B) ⇒ (Q × B → C) ⇒ (P × Q) × A → C
(~~>) :: forall obj (cat :: obj -> obj -> Type) p q a b c .
  ( Monoidal cat
  , Obj cat p, Obj cat q, Obj cat a, Obj cat b, Obj cat c
  , Obj cat (Tensor cat q p)
  , Obj cat (Tensor cat p a)
  , Obj cat (Tensor cat (Tensor cat q p) a)
  , Obj cat (Tensor cat q (Tensor cat p a))
  , Obj cat (Tensor cat q b)
  )
  => cat (Tensor cat p a) b
  -> cat (Tensor cat q b) c
  -> cat (Tensor cat (Tensor cat q p) a) c
-- Note that one could simply write
-- >>> f ~~> g = assocL ~> (id × f) ~> g
-- without a type annotation because of AllowAmbiguousTypes(?).
-- However, to provide a type annotation, we need to make some TypeApplications.
-- See: https://stackoverflow.com/questions/65369389/ for details.
f ~~> g = α ~> (id_q × f) ~> g
  where α    = assocL @_ @_ @q @p @a
        id_q = id @_ @_ @q
