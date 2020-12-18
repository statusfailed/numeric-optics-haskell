{-# LANGUAGE KindSignatures #-}
-- ^ Lets us be explicit about kinds in the type of 'Distributes'
{-# LANGUAGE NoStarIsType   #-}
-- ^ We use Type instead of * in kind signatures
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Numeric.Optics.Types where

import GHC.Types (Type)
import GHC.Exts (Constraint)

-- | Monomorphic lenses over a category and tensor product
data MonoLens tensor cat a b = MonoLens (cat a b) (cat (tensor a b) a)

-- | The type of "context-wrapped" DSL functions.
-- We use this type to provide Category instances for Accelerate.
newtype DSL dsl a b = DSL { runDSL :: dsl a -> dsl b }

-- | Translate between two DSLs
-- For example
-- >>> translate A.map 
-- is a map "lifting" a 'DSL Exp' into a 'DSL Acc' lens by running it on each
-- element of an array.
translate :: ((dsl1 a1 -> dsl1 b1) -> (dsl2 a2 -> dsl2 b2)) -> DSL dsl1 a1 b1 -> DSL dsl2 a2 b2
translate f = DSL . f . runDSL
