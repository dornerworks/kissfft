# Author: Stephen Ng, DornerWorks, Ltd.
# Copyright (c) 2018, DornerWorks, Ltd.
# See LICENSE for full licensing and copying information

CC = gcc

KISSDIR = ../kiss_fft
INCLUDEDIRS = -I$(KISSDIR) -I$(KISSDIR)/tools
WARNINGS = -W -Wall -Wstrict-prototypes -Wmissing-prototypes -Wcast-align \
			-Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast -Wwrite-strings
OPTIMIZATION = -O2

CFLAGS = $(INCLUDEDIRS) $(WARNINGS) $(OPTIMIZATION) -DFIXED_POINT=32 -msse4.1 -mavx2

BUILDDIR = build
TARGET = simd_sample_app
LIBS = -lm -lfftw3

KISSSRC = $(KISSDIR)/kiss_fft_c.c $(KISSDIR)/kiss_fft_sse.c $(KISSDIR)/kiss_fft_avx.c

KISSTOOLSRC = $(KISSDIR)/tools/kiss_fftr_c.c $(KISSDIR)/tools/kiss_fftr_sse.c\
              $(KISSDIR)/tools/kiss_fftr_avx.c

SIMDSRC = print_comparisons.c run_ffts.c simd_sample_app.c

OBJECTS  = $(patsubst %.c,$(BUILDDIR)/%.o,$(SIMDSRC))
OBJECTS += $(patsubst $(KISSDIR)/%.c,$(BUILDDIR)/%.o,$(KISSSRC))
OBJECTS += $(patsubst $(KISSDIR)/tools/%.c,$(BUILDDIR)/%.o,$(KISSTOOLSRC))

DEPS  = $(wildcard *.h)
DEPS += $(wildcard $(KISSDIR)/*.h)
DEPS += $(wildcard $(KISSDIR)/tools/*.h)

$(shell mkdir -p $(BUILDDIR))

.PHONY: default all clean
default: $(BUILDDIR)/$(TARGET)
all: default

$(BUILDDIR)/kiss_fft_c.o: $(KISSDIR)/kiss_fft_c.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(BUILDDIR)/kiss_fft_sse.o: $(KISSDIR)/kiss_fft_sse.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(BUILDDIR)/kiss_fft_avx.o: $(KISSDIR)/kiss_fft_avx.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/kiss_fftr_c.o: $(KISSDIR)/tools/kiss_fftr_c.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(BUILDDIR)/kiss_fftr_sse.o: $(KISSDIR)/tools/kiss_fftr_sse.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
$(BUILDDIR)/kiss_fftr_avx.o: $(KISSDIR)/tools/kiss_fftr_avx.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BUILDDIR) *~ $(TARGET)
