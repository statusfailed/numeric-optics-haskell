{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
module Main where

-- Utilities
import Control.Monad
import System.Random.Shuffle (shuffleM)

-- Accelerate
import Data.Array.Accelerate (Matrix, Vector, Acc, Exp, pattern T2, Z(..), (:.)(..), All(..), slice, lift)
import qualified Data.Array.Accelerate as A

-- Accelerate backends
import qualified Data.Array.Accelerate.LLVM.PTX as Backend

-- numeric optics
import Numeric.Optics (MonoLens(..), runDSL)

-- Iris dataset
import Iris
import Training

main :: IO ()
main = do
  -- Load data
  d <- irisData

  -- Create model and compile SGD inner loop
  let MonoLens f f' = step (gdUpdate (0.01 :: Exp Double)) meanSquaredErrorDisplacement denseModel
      sgdLoop = Backend.runN $ innerSgdLoop (runDSL f')

  params <- mkDenseParams 4 3
  print params

  putStrLn "compiling..."
  let foo = sgdIter $ runDSL f'
  print foo
  let !runEpoch = Backend.runN foo -- (sgdIter $ runDSL f')
  --print runEpoch

  putStrLn "run 100 epochs"
  forM_ [0..100] $ \k -> do
    d' <- shuffleM d
    let a = toAccelerateData $ d'
        b = Backend.run . A.map A.fromIntegral . A.use $ toAccelerateLabelsOneHot d'

    print 0
    --print $ runEpoch a b params
    print $ sgd 150 (sgdLoop a b) params


-------------------------------
-- Training code

sgd :: Int -> (A.Scalar Int -> p -> p) -> p -> p
sgd n go p0 = foldl (\p i -> go (scalar i) p) p0 [0..n - 1]


scalar :: A.Elt i => i -> A.Scalar i
scalar i = A.fromList Z [i]

row :: A.Elt a => Exp Int -> Acc (Matrix a) -> Acc (Vector a)
row i m = A.slice m (lift $ Z :. i :. All)

-- SGD iteration as an Accelerate program
sgdIter :: (A.Elt a, A.Elt b, A.Arrays p)
  => (Acc ((p, Vector a), Vector b) -> Acc (p, Vector a))
  -> Acc (Matrix a)
  -> Acc (Matrix b)
  -> Acc p
  -> Acc p
sgdIter f as bs p = iterateN n (innerSgdLoop f as bs) p where
  (A.I2 n _) = A.shape as

innerSgdLoop :: (A.Elt a, A.Elt b, A.Arrays p)
  => (Acc ((p, Vector a), Vector b) -> Acc (p, Vector a))
  -> Acc (Matrix a)
  -> Acc (Matrix b)
  -> Acc (A.Scalar Int)
  -> Acc p
  -> Acc p
innerSgdLoop f as bs i p = A.afst $ f (T2 (T2 p a) b) where
  a = row (A.the i) as
  b = row (A.the i) bs
