// Copyright 2023 Yunlong Feng
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

#ifndef ARMOR_DETECTOR_OPENVINO__TYPES_HPP_
#define ARMOR_DETECTOR_OPENVINO__TYPES_HPP_

#include <numeric>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>


#include "opencv2/core/mat.hpp"


namespace rm_auto_aim
{
  struct Light : public cv::RotatedRect {
    Light() = default;
    explicit Light(const std::vector<cv::Point> &contour)
    : cv::RotatedRect(cv::minAreaRect(contour)) {
     
  
      center = std::accumulate(
        contour.begin(),
        contour.end(),
        cv::Point2f(0, 0),
        [n = static_cast<float>(contour.size())](const cv::Point2f &a, const cv::Point &b) {
          return a + cv::Point2f(b.x, b.y) / n;
        });
  
      cv::Point2f p[4];
      this->points(p);
      std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
      top = (p[0] + p[1]) / 2;
      bottom = (p[2] + p[3]) / 2;


  
      length = cv::norm(top - bottom);
      width = cv::norm(p[0] - p[1]);
  
      axis = top - bottom;
      axis = axis / cv::norm(axis);
  
      // Calculate the tilt angle
      // The angle is the angle between the light bar and the horizontal line
      tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
      tilt_angle = tilt_angle / CV_PI * 180;
    }
   
    cv::Point2f top, bottom, center;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;
  };
  struct LightParams {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
   
  };

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE };

typedef struct ArmorObject
{
  ArmorColor color;
  ArmorNumber number;
  float prob;
  std::vector<cv::Point2f> pts;
  std::vector<cv::Point2f> pts_binary;
  cv::Rect box;


  cv::Mat number_img;  

  double confidence;   

  cv::Mat whole_binary_img;
  cv::Mat whole_rgb_img;
  cv::Mat whole_gray_img;

  std::vector<Light> lights;

  double new_x;
  double new_y;
  bool is_ok =false;

  //std::unique_ptr<LightCornerCorrector> corner_corrector;

} ArmorObject;

constexpr const char * K_ARMOR_NAMES[] = {"guard", "1", "2", "3", "4", "5", "outpost", "base"};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_OPENVINO__TYPES_HPP_
