#pragma once

#include <cstdint>

int getPhysicalCoreCount();

int64_t getSystemRAMSize();

double getSystemRAMSizeGB();

double getTotalRAMUsageGB();

static const unsigned int PHYSICAL_CORE_COUNT = (unsigned int)getPhysicalCoreCount();

static const int64_t RAM_IN_BYTES = getSystemRAMSize();

static const double RAM_IN_GB = getSystemRAMSizeGB();