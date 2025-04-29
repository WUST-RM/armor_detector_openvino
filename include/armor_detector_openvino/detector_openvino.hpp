// Copyright 2023 Yunlong Feng
// Copyright 2025 Lihan Chen
// Copyright 2025 XiaoJian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ARMOR_DETECTOR_OPENVINO__DETECTOR_OPENVINO_HPP_
#define ARMOR_DETECTOR_OPENVINO__DETECTOR_OPENVINO_HPP_

#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "armor_detector_openvino/light_corner_corrector.hpp"
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "armor_detector_openvino/ThreadPool.h"

namespace rm_auto_aim
{
  

class DetectorOpenVino
{
public:
  using DetectorCallback =
    std::function<void(const std::vector<ArmorObject> &, int64_t, const cv::Mat &)>;

public:
  /**
   * @brief Construct a new OpenVINO Detector object
   *
   * @param model_path IR/ONNX file path
   * @param device_name Target device (CPU, GPU, AUTO)
   * @param conf_threshold Confidence threshold for output filtering
   * @param top_k Topk parameter
   * @param nms_threshold NMS threshold
   * @param auto_init If initializing detector inplace
   */
  explicit DetectorOpenVino(
    const std::filesystem::path & model_path, const std::string & classify_model_pathconst,
    std::string & classify_label_path, const std::string & device_name,const LightParams &l , float conf_threshold = 0.25,
    int top_k = 128, float nms_threshold = 0.3, float expand_ratio_w= 2.0f, float expand_ratio_h= 1.5f, int binary_thres_= 85, bool auto_init = false);

  /**
   * @brief Initialize detector
   *
   */
  void init();
  /**
   * @brief Push a single image to inference
   *
   * @param rgb_img
   * @param timestamp_nanosec
   * @return std::future<bool> If callback finished return true.
   */
  //std::future<bool> pushInput(const cv::Mat & rgb_img, int64_t timestamp_nanosec);
  void pushInput(const cv::Mat& rgb_img, int64_t timestamp_nanosec);
  /**
   * @brief Set the inference callback
   *
   * @param callback
   */
  void setCallback(DetectorCallback callback);

  std::vector<Light> findLights(const cv::Mat &rbg_img,
    const cv::Mat &binary_img,ArmorObject & armor) noexcept;

  bool isLight(const Light &possible_light) noexcept;

  void detect(ArmorObject & armor);

  

private:
  bool processCallback(
    const cv::Mat resized_img, Eigen::Matrix3f transform_matrix, int64_t timestamp_nanosec,
    const cv::Mat & src_img);

  void initNumberClassifier();

  void extractNumberImage(const cv::Mat & src, ArmorObject & armor);

  bool classifyNumber(ArmorObject & armor);

  LightParams light_params_;

private:


  std::string model_path_;
  std::string classify_model_path_;
  std::string classify_label_path_;
  std::string device_name_;
  float conf_threshold_;
  int top_k_;
  float nms_threshold_;
  std::vector<int> strides_;
  std::vector<GridAndStride> grid_strides_;

  DetectorCallback infer_callback_;

  std::unique_ptr<ov::Core> ov_core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;



  cv::dnn::Net number_net_;
  std::vector<std::string> class_names_;
  float number_threshold_;
  int binary_thres_ ;


  float expand_ratio_w_ ;
  float expand_ratio_h_ ;

  std::vector<Light> lights_;



  std::unique_ptr<ThreadPool> thread_pool_;


};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_OPENVINO__DETECTOR_OPENVINO_HPP_
