#ifndef LITETORCH_HUB_H
#define LITETORCH_HUB_H

#include "litetorch/tensor.h"
#include <map>
#include <memory>
#include <string>

namespace litetorch {
namespace hub {

std::string get_dir();
void set_dir(const std::string& dir);
bool is_url(const std::string& path);
std::string filename_from_url(const std::string& url);
std::string download_url_to_file(const std::string& url, const std::string& dst = "",
                                 bool show_progress = true);
std::map<std::string, std::shared_ptr<Tensor>> load_state_dict_from_url(
    const std::string& url, bool force_download = false);

}
}

#endif
