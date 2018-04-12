/**
 * \file print_comparisons.c
 *
 * \brief Compares and formats messages describing each FFT result
 * \details See README
 *
 * \copyright Copyright (c) 2018, DornerWorks, Ltd. 
 * \license See LICENSE for full licensing and copying information
 */


/******************************************************************************
 *                                                                 Inclusions *
 ******************************************************************************/
#include "print_comparisons.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>


/******************************************************************************
 *                                                                    Defines *
 ******************************************************************************/
#define MAX_ERROR_PRINTS (100)
#define PRINT_CORRECT_VALS (false)


/******************************************************************************
 *                                                      Function declarations *
 ******************************************************************************/
static void printSingleFftXComparison(kiss_fft_cpx *master, kiss_fft_cpx *other, int32_t len,
                                      verboseness_t verbose);


/******************************************************************************
 *                                                         External functions *
 ******************************************************************************/
void printFftComparisons(results_t master_results, results_t other_results[NUM_FFT_COMPARISONS])
{
   int i;
   printf("=== FFT Comparison Results ===\n");
   printf("Master results are from %s FFT\n", master_results.name);
   if(ALLOWABLE_ERROR)
   {
      printf("(allowable error is %d)\n", ALLOWABLE_ERROR);
   }

   for(i = 0; i < NUM_FFT_COMPARISONS; i++)
   {
      printf("\n%s results:\n", other_results[i].name);

      float runtimePct = 100.0f * other_results[i].runtime_ns / master_results.runtime_ns;
      printf("Runtime as a percentage of baseline: %1.2f%%\n", runtimePct);
      printf("Accuracy:\n");
      printSingleFftXComparison((kiss_fft_cpx *)*master_results.output.pfft,
                                (kiss_fft_cpx *)*other_results[i].output.pfft,
                                FFT_LEN,
                                other_results[i].verbose);
   }
   printf("\n\n");
}

void printFftrComparisons(results_t master_results, results_t other_results[NUM_FFT_COMPARISONS])
{
   int i;
   printf("=== FFTR Comparison Results ===\n");
   printf("Baseline results are from %s FFTR\n", master_results.name);
   if(ALLOWABLE_ERROR)
   {
      printf("(allowable error is %d)\n", ALLOWABLE_ERROR);
   }

   for(i = 0; i < NUM_FFT_COMPARISONS; i++)
   {
      printf("\n%s results:\n", other_results[i].name);

      float runtimePct = 100.0f * other_results[i].runtime_ns / master_results.runtime_ns;
      printf("Runtime as a percentage of baseline: %1.2f%%\n", runtimePct);
      printf("Accuracy:\n");
      printSingleFftXComparison((kiss_fft_cpx *)*master_results.output.pfftr,
                                (kiss_fft_cpx *)*other_results[i].output.pfftr,
                                FFTR_OUT_LEN,
                                other_results[i].verbose);
   }
   printf("\n\n");
}


/******************************************************************************
 *                                                         Internal functions *
 ******************************************************************************/
static void printSingleFftXComparison(kiss_fft_cpx *master, kiss_fft_cpx *other, int32_t len,
                                      verboseness_t verbose)
{
   int32_t num_wrong = 0;
   int32_t abs_error = 0;
   int32_t net_error = 0;
   int32_t num_zero_values = 0;
   float pct_wrong, pct_abs_error, pct_net_error, avg_value, avg_abs_error, avg_net_error;
   int32_t sum_total = 0;

   int16_t m_re, m_im, o_re, o_im;
   int16_t *comparing[2][2] = {
      {&m_re, &o_re},
      {&m_im, &o_im},
   };
   const char *names[2] = {
      "real",
      "imag",
   };

   for(int32_t i = 0; i < len; i++)
   {
      m_re = master[i].r;
      m_im = master[i].i;
      o_re = other[i].r;
      o_im = other[i].i;

      sum_total += abs(m_re) + abs(m_im);
      num_zero_values += (0 == m_re) ? 1 : 0;
      num_zero_values += (0 == m_im) ? 1 : 0;

      for(int32_t j = 0; j < 2; j++)
      {
         int32_t master_val = *(comparing[j][0]);
         int32_t other_val  = *(comparing[j][1]);
         if (ALLOWABLE_ERROR < abs(other_val - master_val))
         {
            num_wrong++;
            abs_error += abs(other_val - master_val);
            net_error += other_val - master_val;
            if(verbose && MAX_ERROR_PRINTS >= num_wrong)
            {
               printf("MISMATCH %04u: %s: orig %5d, got %5d\n", i, names[j], master_val, other_val);
               if (MAX_ERROR_PRINTS == num_wrong)
               {
                  printf("Not printing any more entries...\n\n");
               }
            }
         }
         else if (PRINT_CORRECT_VALS)
         {
            if (verbose && MAX_ERROR_PRINTS > num_wrong)
            {
               printf("match %04u: %s: %5d\n", i, names[j], other_val);
            }
         }
      }
   }

   pct_wrong = 100.0f*(float)num_wrong/(float)(len*2);
   avg_abs_error = (float)abs_error/(float)num_wrong;
   avg_net_error = (float)net_error/(float)num_wrong;
   avg_value = (float)sum_total/(float)((len*2) - num_zero_values);
   pct_abs_error = 100.0f*avg_abs_error/avg_value;
   pct_net_error = 100.0f*fabs(avg_net_error/avg_value);
   if(0 == num_wrong)
   {
      printf("\tMATCH\n");
   }
   else
   {
      printf("\tnum different:  %d (%1.2f%%)\n", num_wrong, pct_wrong);
      printf("\tavg abs error:  %1.2f (%1.2f%%)\n", avg_abs_error, pct_abs_error);
      printf("\tavg net error:  %1.2f (%1.2f%%)\n", avg_net_error, pct_net_error);
   }
}

