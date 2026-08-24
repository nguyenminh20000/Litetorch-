#ifndef LITETORCH_TPU_H
#define LITETORCH_TPU_H

#include "litetorch/device.h"
#include <string>

namespace litetorch {
namespace tpu {

bool is_available();
int device_count();
int current_device();
void set_device(int device_id);
void synchronize();
std::string get_device_name(int device_id = 0);

}
}

#endif
