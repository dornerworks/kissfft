#!/bin/bash

WARNINGS='-W -Wall -Wstrict-prototypes -Wmissing-prototypes -Waggregate-return
    -Wcast-align -Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast
    -Wwrite-strings'

OPTIMIZATION='-O2'

CFLAGS='-DFIXED_POINT=32'

SRCDIR=.
KISSDIR=../kiss_fft
INCLUDEDIRS="-I$KISSDIR -I$KISSDIR/tools"
BUILDDIR=./build
TARGET=simd_sample_app

mkdir -p $BUILDDIR

# Compile KISS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/kiss_fft_c.c -o $BUILDDIR/kiss_fft_c.o $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/kiss_fft_sse.c -o $BUILDDIR/kiss_fft_sse.o -msse4 $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/kiss_fft_avx.c -o $BUILDDIR/kiss_fft_avx.o -mavx2 $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/tools/kiss_fftr_c.c -o $BUILDDIR/kiss_fftr_c.o $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/tools/kiss_fftr_sse.c -o $BUILDDIR/kiss_fftr_sse.o -msse4 $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS $KISSDIR/tools/kiss_fftr_avx.c -o $BUILDDIR/kiss_fftr_avx.o -mavx2 $CFLAGS

# Compile test app
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS print_comparisons.c -o $BUILDDIR/print_comparisons.o $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS run_ffts.c -o $BUILDDIR/run_ffts.o $CFLAGS
gcc -c $WARNINGS $OPTIMIZATION $INCLUDEDIRS simd_sample_app.c -o $BUILDDIR/simd_sample_app.o $CFLAGS

# Link
gcc -o $BUILDDIR/$TARGET $BUILDDIR/*.o -lm -lfftw3

