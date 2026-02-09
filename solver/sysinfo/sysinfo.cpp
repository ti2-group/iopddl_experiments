#include "sysinfo.h"
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <vector>
#elif defined(__linux__)
#include <fstream>
#include <set>
#include <string>
#include <sys/sysinfo.h>
#elif defined(__APPLE__)

#include <sys/sysctl.h>
#include <sys/types.h>

#endif


#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
int getPhysicalCoreCount() {
  DWORD len = 0;
  GetLogicalProcessorInformation(nullptr, &len);
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  GetLogicalProcessorInformation(buffer.data(), &len);

  int coreCount = 0;
  for (const auto& info : buffer) {
    if (info.Relationship == RelationProcessorCore) {
      coreCount++;
    }
  }
  if(coreCount < 1){
      return int(std::thread::hardware_concurrency());
  }
  return coreCount;
}
#elif defined(__linux__)
int getPhysicalCoreCount() {
    char buffer[128];
    const char* cmd = "lscpu -p=core,socket | grep -v '#' | sort -u | wc -l";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return int(std::thread::hardware_concurrency());
    }

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        pclose(pipe);

        int coreCount = std::atoi(buffer);
        if(coreCount < 1){
            return int(std::thread::hardware_concurrency());
        }
        return coreCount;
    }

    pclose(pipe);
    return int(std::thread::hardware_concurrency());
}
#elif defined(__APPLE__)

int getPhysicalCoreCount() {
    int coreCount;
    size_t len = sizeof(coreCount);
    sysctlbyname("hw.physicalcpu", &coreCount, &len, nullptr, 0);

    if (coreCount < 1) {
        return int(std::thread::hardware_concurrency());
    }
    return coreCount;
}

#else
int getPhysicalCoreCount() {
  return int(std::thread::hardware_concurrency());
}
#endif

#ifdef __linux__
#include <sys/sysinfo.h>
double getTotalRAMUsageGB() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        double used = (info.totalram - info.freeram) * info.mem_unit;
        return used / (1024.0 * 1024.0 * 1024.0); // convert to GB
    }
    return -1;
}
#elif __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
double getTotalRAMUsageGB() {
  vm_size_t page_size;
  mach_port_t mach_port = mach_host_self();
  vm_statistics64_data_t vm_stats;
  mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);

  if (host_page_size(mach_port, &page_size) != KERN_SUCCESS ||
      host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) != KERN_SUCCESS) {
    return -1;
  }

  uint64_t used_memory = (vm_stats.active_count + vm_stats.inactive_count + vm_stats.wire_count) * page_size;
  return used_memory / (1024.0 * 1024.0 * 1024.0); // convert to GB
}
#else
double getTotalRAMUsageGB() {
    return -1; // unsupported OS
}
#endif

int64_t getSystemRAMSize() {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return static_cast<int64_t>(statex.ullTotalPhys);
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
      return static_cast<int64_t>(info.totalram) * info.mem_unit;
    } else {
      return -1; // error case
    }
#elif defined(__APPLE__)
    int mib[2];
    int64_t physical_memory;
    size_t length;
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    return physical_memory;
#else
#error "Unknown platform or unsupported OS"
#endif
}

double getSystemRAMSizeGB() {
#if defined(_WIN32) || defined(_WIN64)
  MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return static_cast<double>(statex.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    } else {
        return -1.0; // error case
    }
#elif defined(__linux__)
  struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<double>(info.totalram) * info.mem_unit / (1024.0 * 1024.0 * 1024.0);
    } else {
        return -1.0; // error case
    }
#elif defined(__APPLE__)
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  int64_t physical_memory;
  size_t length = sizeof(physical_memory);
  if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
    return static_cast<double>(physical_memory) / (1024.0 * 1024.0 * 1024.0);
  } else {
    return -1.0; // error case
  }
#else
#error "Unknown platform or unsupported OS"
#endif
}

