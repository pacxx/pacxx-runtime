//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/cuda/CUPTIProfiler.h"
#include "pacxx/Executor.h"
#include "pacxx/detail/common/jsonHelper.h"
#include "pacxx/detail/cuda/CUDARuntime.h"
#include <fstream>

namespace pacxx {
namespace v2 {

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

uint64_t CUPTIProfiler::kernelDuration = 0;

CUPTIProfiler::CUPTIProfiler() : stage(0), metricId(0), passData(nullptr) {}

bool CUPTIProfiler::checkStage(unsigned required, std::string stagename,
                               bool minonly) {
  if (!enabled())
    return false;
  if (!minonly && (required < stage)) {
    __error("CUPTIProfiler is already past stage ", required, ", can't run ",
            stagename);
    return false;
  } else if (required > stage) {
    __error("CUPTIProfiler is not yet ready for stage ", required,
            ", can't run ", stagename);
    return false;
  } else
    return true;
}

bool CUPTIProfiler::preinit(void *settings) {
  if (!checkStage(0, "preinit"))
    return false;
  __verbose("CUPTIProfiler preinit");
  SEC_CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  stage = 1;
  return true;
}

bool CUPTIProfiler::postinit(void *settings) {
  if (!checkStage(1, "postinit"))
    return false;
  __verbose("CUPTIProfiler postinit");
  SEC_CUDA_CALL(cuDeviceGet(
      &_device, *(static_cast<unsigned *>(settings)))); /// this is broken in
                                                        /// sooo many ways (what
                                                        /// happens with "old"
                                                        /// context?)
  SEC_CUPTI_CALL(
      cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  std::ifstream profiles(InFilePath);
  if (profiles.is_open()) {
    for (std::string metricString; std::getline(profiles, metricString);)
      profilingMetrics.push_back(metricString);
    profiles.close();
  }
  stage = 2;
  return true;
}

void CUPTIProfiler::updateKernel(Kernel *kernel) {
  if (!checkStage(2, "updateKernel", true))
    return;
  __verbose("CUPTIProfiler updateKernel");
  _kernel = kernel;
  stats[_kernel->getName()].emplace_back();
  stats[_kernel->getName()].back().first = _kernel->getConfiguration();
  __verbose("Current kernel run count: ", stats[_kernel->getName()].size());
  SEC_CUPTI_CALL(
      cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  stage = 3;
}

void CUPTIProfiler::dryrun() {
  if (!checkStage(3, "dryrun"))
    return;
  __verbose("CUPTIProfiler dryrun");
  _kernel->getRuntime().synchronize();
  SEC_CUPTI_CALL(cuptiActivityFlushAll(0));
  _kernel->launch();
  Executor::get().restoreArgs();
  _kernel->getRuntime().synchronize();
  SEC_CUPTI_CALL(cuptiActivityFlushAll(0));
  {
    std::stringstream foo;
    foo << kernelDuration;
    foo << "us";
    stats[_kernel->getName()].back().second["kernelDuration"] = foo.str();
  }
  __debug("Kernel run time: ", kernelDuration, "us");
  stage = 4;
}

void CUPTIProfiler::profile() {
  if (!checkStage(4, "profile"))
    return;
  __verbose("CUPTIProfiler profile");
  for (const std::string &metricString : profilingMetrics)
    profileSingle(metricString);
}

void CUPTIProfiler::report() {
  nlohmann::json rprt(stats);
  if (OutFilePath.empty())
    __message(rprt.dump(4));
  else {
    std::ofstream ReportFile(OutFilePath,
                             std::ios_base::out | std::ios_base::trunc);
    if (ReportFile.is_open()) {
      ReportFile << std::setw(4) << rprt << std::endl;
      ReportFile.close();
    }
  }
}

void CUPTIAPI CUPTIProfiler::bufferRequested(uint8_t **buffer, size_t *size,
                                             size_t *maxNumRecords) {
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    __error("Error: out of memory\n");
    exit(-1);
  }
}

void CUPTIAPI CUPTIProfiler::bufferCompleted(CUcontext ctx, uint32_t streamId,
                                             uint8_t *buffer, size_t size,
                                             size_t validSize) {
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel3 *kernel;

  // since we launched only 1 kernel, we should have only 1 kernel record
  SEC_CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel3 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    __error("Error: expected kernel activity record, got ", (int)kernel->kind,
            "\n");
    exit(-1);
  }

  kernelDuration = kernel->end - kernel->start;
  free(buffer);
}

void CUPTIAPI CUPTIProfiler::getMetricValueCallback(
    void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbInfo) {
  MetricData_t *metricData = (MetricData_t *)userdata;
  unsigned int i, j, k;

  // This callback is enabled only for launch so we shouldn't see
  // anything else.
  /// MOD FOR NVTX
  if (domain != 1 || cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    __error("unexpected cbid ", cbid);
    return;
  }

  // on entry, enable all the event groups being collected this pass,
  // for metrics we collect for all instances of the event
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    _kernel->getRuntime().synchronize();

    SEC_CUPTI_CALL(cuptiSetEventCollectionMode(
        cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      uint32_t all = 1;
      SEC_CUPTI_CALL(cuptiEventGroupSetAttribute(
          metricData->eventGroups->eventGroups[i],
          CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all),
          &all));
      SEC_CUPTI_CALL(
          cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
    }
  }

  // on exit, read and record event values
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    _kernel->getRuntime().synchronize();

    // for each group, read the event values from the group and record
    // in metricData
    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
      CUpti_EventDomainID groupDomain;
      uint32_t numEvents, numInstances, numTotalInstances;
      CUpti_EventID *eventIds;
      size_t groupDomainSize = sizeof(groupDomain);
      size_t numEventsSize = sizeof(numEvents);
      size_t numInstancesSize = sizeof(numInstances);
      size_t numTotalInstancesSize = sizeof(numTotalInstances);
      uint64_t *values, normalized, sum;
      size_t valuesSize, eventIdsSize;

      SEC_CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize,
          &groupDomain));
      SEC_CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
          metricData->device, groupDomain,
          CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize,
          &numTotalInstances));
      SEC_CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstancesSize,
          &numInstances));
      SEC_CUPTI_CALL(
          cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                      &numEventsSize, &numEvents));
      eventIdsSize = numEvents * sizeof(CUpti_EventID);
      eventIds = (CUpti_EventID *)malloc(eventIdsSize);
      SEC_CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds));

      valuesSize = sizeof(uint64_t) * numInstances;
      values = (uint64_t *)malloc(valuesSize);

      for (j = 0; j < numEvents; j++) {
        SEC_CUPTI_CALL(
            cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                     eventIds[j], &valuesSize, values));
        if (metricData->eventIdx >= metricData->numEvents) {
          __error("too many events collected, metric expects only ",
                  (int)metricData->numEvents);
          return;
        }

        // sum collect event values from all instances
        sum = 0;
        for (k = 0; k < numInstances; k++)
          sum += values[k];

        // normalize the event value to represent the total number of
        // domain instances on the device
        normalized = (sum * numTotalInstances) / numInstances;

        metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
        metricData->eventValueArray[metricData->eventIdx] = normalized;
        metricData->eventIdx++;

        // print collected value
        {
          std::stringstream eventlog("");
          char eventName[128];
          size_t eventNameSize = sizeof(eventName) - 1;
          SEC_CUPTI_CALL(cuptiEventGetAttribute(
              eventIds[j], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
          eventName[127] = '\0';
          eventlog << "\t" << eventName << " = " << (unsigned long long)sum
                   << " (";
          if (numInstances > 1) {
            for (k = 0; k < numInstances; k++) {
              if (k != 0)
                eventlog << ", ";
              eventlog << (unsigned long long)values[k];
            }
          }

          eventlog << ")\n";
          eventlog << "\t" << eventName << "(normalized) ("
                   << (unsigned long long)sum << " * " << numTotalInstances
                   << ") / " << numInstances << " = "
                   << (unsigned long long)normalized;
          __verbose(eventlog.str());
        }
      }

      free(values);
    }

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
      SEC_CUPTI_CALL(
          cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
  }
}

void CUPTIProfiler::profileSingle(const std::string &metricName) {
  try {
    SEC_CUPTI_CALL(
        cuptiMetricGetIdFromName(_device, metricName.c_str(), &metricId));
  } catch (common::generic_exception ex) {
    __debug("Metric name ", metricName, " is invalid.");
    return;
  }
  __verbose("Metric id is ", metricId);
  // setup launch callback for event collection
  SEC_CUPTI_CALL(cuptiSubscribe(
      &subscriber, (CUpti_CallbackFunc)CUPTIProfiler::getMetricValueCallback,
      &metricData));

  SEC_CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                     CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));

  // allocate space to hold all the events needed for the metric
  SEC_CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
  metricData.device = _device;
  metricData.eventIdArray =
      (CUpti_EventID *)malloc(metricData.numEvents * sizeof(CUpti_EventID));
  metricData.eventValueArray =
      (uint64_t *)malloc(metricData.numEvents * sizeof(uint64_t));
  metricData.eventIdx = 0;

  // get the number of passes required to collect all the events
  // needed for the metric and the event groups for each pass
  SEC_CUPTI_CALL(cuptiMetricCreateEventGroupSets(
      static_cast<CUDARuntime &>(_kernel->getRuntime()).getContext(),
      sizeof(metricId), &metricId, &passData));
  __debug("CUPTI requested ", passData->numSets, " kernel passes.");
  for (uint32_t pass = 0; pass < passData->numSets; pass++) {
    __verbose("Pass ", pass);
    metricData.eventGroups = passData->sets + pass;
    _kernel->launch();
    Executor::get().restoreArgs();
  }

  if (metricData.eventIdx != metricData.numEvents) {
    __error("expected ", metricData.numEvents, " metric events, got ",
            metricData.eventIdx);
  } else {
    // use all the collected events to calculate the metric value
    SEC_CUPTI_CALL(cuptiMetricGetValue(
        _device, metricId, metricData.numEvents * sizeof(CUpti_EventID),
        metricData.eventIdArray, metricData.numEvents * sizeof(uint64_t),
        metricData.eventValueArray, kernelDuration, &metricValue));

    // store metric value
    {
      CUpti_MetricValueKind valueKind;
      size_t valueKindSize = sizeof(valueKind);
      SEC_CUPTI_CALL(cuptiMetricGetAttribute(
          metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
      stats[_kernel->getName()].back().second[metricName] =
          stringSingleMetric(std::make_pair(metricValue, valueKind));
      //__message("Metric ", metricName, " recorded.");
    }
  }
  SEC_CUPTI_CALL(cuptiUnsubscribe(subscriber));
}

std::string CUPTIProfiler::stringSingleMetric(
    const std::pair<CUpti_MetricValue, CUpti_MetricValueKind> &entry) {
  std::stringstream buffer;
  switch (entry.second) {
  case CUPTI_METRIC_VALUE_KIND_DOUBLE:
    buffer << entry.first.metricValueDouble;
    break;
  case CUPTI_METRIC_VALUE_KIND_UINT64:
    buffer << (unsigned long long)entry.first.metricValueUint64;
    break;
  case CUPTI_METRIC_VALUE_KIND_INT64:
    buffer << (long long)entry.first.metricValueInt64;
    break;
  case CUPTI_METRIC_VALUE_KIND_PERCENT:
    buffer << entry.first.metricValuePercent;
    buffer << "%";
    break;
  case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
    buffer << (unsigned long long)entry.first.metricValueThroughput;
    buffer << "b/s";
    break;
  case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
    buffer << "ul ";
    buffer << (unsigned int)entry.first.metricValueUtilizationLevel;
    break;
  default:
    buffer << "unknown value kind";
  }
  return buffer.str();
}
}
} // namespace pacxx
