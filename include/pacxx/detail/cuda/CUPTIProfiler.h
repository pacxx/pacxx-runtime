//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_CUPTIPROFILER_H
#define PACXX_V2_CUPTIPROFILER_H

#include "../IRProfiler.h"
#include "CUDAKernel.h"
#include "CUDAErrorDetection.h"
#include <list>

namespace pacxx {
namespace v2 {

class CUPTIProfiler : public IRProfiler {
public:
  typedef struct MetricData_st {
	  // the device where metric is being collected
	  CUdevice device;
	  // the set of event groups to collect for a pass
	  CUpti_EventGroupSet *eventGroups;
	  // the current number of events collected in eventIdArray and
	  // eventValueArray
	  uint32_t eventIdx;
	  // the number of entries in eventIdArray and eventValueArray
	  uint32_t numEvents;
	  // array of event ids
	  CUpti_EventID *eventIdArray;
	  // array of event values
	  uint64_t *eventValueArray;
  } MetricData_t;

  CUPTIProfiler();

  virtual ~CUPTIProfiler() {};

  virtual bool preinit(void* settings) override;

  virtual void postinit(void* settings) override;

  virtual void updateKernel(Kernel *kernel) override;

  virtual void dryrun() override;

  virtual void profile() override;

  virtual void report() override;

private:
  bool checkStage(unsigned required, std::string stagename, bool minonly = false);
  static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
  static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
  static void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);
  void profileSingle(const std::string& metricName);
  std::string stringSingleMetric(const std::pair<CUpti_MetricValue, CUpti_MetricValueKind>& entry);

  CUdevice _device;
  unsigned stage;
  static uint64_t kernelDuration;
  CUpti_SubscriberHandle subscriber;
  CUpti_MetricID metricId;
  MetricData_t metricData;
  CUpti_EventGroupSets *passData;
  CUpti_MetricValue metricValue;
  std::list<std::string> profilingMetrics;

  ///      Kernel name, run count,           metric name,                metric value, metric value kind
  //std::map<std::string, std::vector<std::map<std::string, std::pair<CUpti_MetricValue, CUpti_MetricValueKind>>>> stats;
  ///      Kernel name, run count,           metric name, stringified value
  std::map<std::string, std::vector<std::map<std::string, std::string>>> stats;
};
}
}

#endif // PACXX_V2_CUPTIPROFILER_H
