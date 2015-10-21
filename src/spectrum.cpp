/**
 * This file is part of the point set analysis tool psa
 *
 * Copyright 2012
 * Thomas Schlömer, thomas.schloemer@uni-konstanz.de
 * Daniel Heck, daniel.heck@uni-konstanz.de
 *
 * psa is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "spectrum.h"
#include "util.h"
#include <cmath>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif


Spectrum::Spectrum(int size) {
    this->size = size;
    this->ft = new float[size * size * 2];
    for (int i = 0; i < size * size * 2; ++i)
        ft[i] = 0;
}

Spectrum& Spectrum::operator= (const Spectrum &s) {
    this->size = s.size;
    if (this->ft) delete this->ft;
    this->ft = new float[size * size * 2];
    memcpy(this->ft, s.ft, size * size * 2 * sizeof(float));
    return *this;
}

#define PIONEOVER 0.3183098861837907f
#define PIONE_TWO_OVER 0.15915494309189535

static inline float cosf3(float f) {
  float sgn = (f >= 0.0f)*2.0-1.0;
  
  f *= sgn*PIONE_TWO_OVER;
  
  float fract = f - (float)((int)f);
  fract = sgn < 0.0f ? 1.0f - fract : fract;
  
  float quad = (float)((int)(fract*2.0f));
  
  fract = fract*2.0f - quad;
  fract = quad < 0.5f ? (fract-0.5)*2.0f : fract - 1.0;
  
  return fract;
}

static inline float cosf2(float f) {
  float sgn = (f >= 0.0f)*2.0-1.0;
  
  f *= sgn*PIONE_TWO_OVER;
  
  f -= (float)((int)f);
  
  f = sgn < 0.0 ? 1.0 - f : f;
  
  f = (f-0.5)*2.0;
  f *= (f >= 0.0)*2.0f - 1.0f;
  
  //s-power series

  //level one is degree-3 smoothstep
  //f = f*f*(3.0-2.0*f);
  
  //level two;
  //f = (-((s-1+s)*(M_PI*M_PI-12)*(s-1)*(s-1)+4*(2*s-3))*s*s)/4.0;
  
  //level three:
  f = (-((f-1+f)*(M_PI*M_PI-12)*(f-1)*(f-1)+
      4*(2*f-3)-4*(f-1+f)*(M_PI*M_PI-10)*(f-1)*(f-1)*(f-1)*f)*f*f)/4.0;
  
  f = (f-0.5)*2.0;
  
  return f;
}

#define PIOVER2 1.5707963267948966f

static inline float sinf2(float f) {
  return cosf2(f+PIOVER2);
}

void Spectrum::PointSetSpectrum(Spectrum *spectrum, const PointSet &points,
                                const int npoints)
{
    const int size2 = spectrum->size / 2;
#ifdef _OPENMP
#pragma omp parallel
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    //printf("%f %f %f %f %f\n", cosf2(0.0), cosf2(M_PI), cosf2(M_PI*2.0), cosf2(M_PI*0.9), cosf2(M_PI*0.5));
    
    for (int x = 0; x < spectrum->size; ++x) {
        if (x % 2) {
          printf("doing row %d of %d\r", x+1, spectrum->size);
          fflush(stdout);
        }
        
        for (int y = 0; y < spectrum->size; ++y) {
            float fx = 0.f, fy = 0.f;
            float wx = x - size2;
            float wy = y - size2;
            
            for (int i = 0; i < npoints; ++i) {
                float exp = -TWOPI * (wx * points[i].x + wy * points[i].y);
                
                fx += cosf2(exp);
                fy += sinf2(exp);
            }
            spectrum->ft[2*(x + y*spectrum->size)  ] = fx;
            spectrum->ft[2*(x + y*spectrum->size)+1] = fy;
        }
    }
    
    //eat final \r
    printf("\n");
}
}

