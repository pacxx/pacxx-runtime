//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include </usr/include/math.h> // FIXME: PACXX gets confused with cmath from libc++ on some platforms thats
                               //        why we include math.h with an absolute path here

#include "pacxx/detail/device/DeviceCode.h"
#include "pacxx/detail/device/DeviceFunctionDecls.h"


extern "C" {
	double rsqrt(double val) {
		return 1.0 / sqrt(val);
	}

    float rsqrtf(float val) {
		return 1.0f / sqrtf(val);
	}
}