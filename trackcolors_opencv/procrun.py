import tryingMultiprocessing

if __name__ == '__main__':
    extractor = tryingMultiprocessing.ParallelExtractor()  #not mine doesn't work

    extractor.runInParallel(numProcesses=2, numThreads=4)