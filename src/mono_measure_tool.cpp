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

#include <armor_detector_openvino/mono_measure_tool.hpp>
#include <iostream>

namespace rm_auto_aim
{

std::vector<cv::Point3f> MonoMeasureTool::SMALL_ARMOR_3D_POINTS = {
  {0, 0.027, -0.066},
  {0, -0.027, -0.066},
  {0, -0.027, 0.066},
  {0, 0.027, 0.066},
};

std::vector<cv::Point3f> MonoMeasureTool::BIG_ARMOR_3D_POINTS = {
  {0, 0.027, -0.1125},
  {0, -0.027, -0.1125},
  {0, -0.027, 0.1125},
  {0, 0.027, 0.1125},
};

bool is_big_armor(const ArmorObject & obj)
{
  switch (obj.number) {
    case ArmorNumber::NO1:
    case ArmorNumber::BASE:
      return true;
    default:
      return false;
  }
}

MonoMeasureTool::MonoMeasureTool(
  std::vector<double> camera_intrinsic, std::vector<double> camera_distortion)
{
  setCameraInfo(camera_intrinsic, camera_distortion);
}

bool MonoMeasureTool::setCameraInfo(
  std::vector<double> camera_intrinsic, std::vector<double> camera_distortion)
{
  if (camera_intrinsic.size() != 9) {
    // the size of camera intrinsic must be 9 (equal 3*3)
    return false;
  }
  // init camera_intrinsic and camera_distortion
  cv::Mat camera_intrinsic_mat(camera_intrinsic, true);
  camera_intrinsic_mat = camera_intrinsic_mat.reshape(0, 3);
  camera_intrinsic_ = camera_intrinsic_mat.clone();

  cv::Mat camera_distortion_mat(camera_distortion, true);
  camera_distortion_mat = camera_distortion_mat.reshape(0, 1);
  camera_distortion_ = camera_distortion_mat.clone();
  return true;
}

bool MonoMeasureTool::solvePnp(
  const std::vector<cv::Point2f> & points2d, const std::vector<cv::Point3f> & points3d,
  cv::Point3f & position, cv::Mat & rvec, cv::SolvePnPMethod pnp_method)
{
  if (points2d.size() != points3d.size()) {
    return false;  // 投影点数量不匹配
  }
  // cv::Mat rot = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat trans = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat r;  // 旋转向量
  bool res = cv::solvePnP(
    points3d, points2d, camera_intrinsic_, camera_distortion_, r, trans, false, pnp_method);
  rvec = r.clone();
  position = cv::Point3f(trans);
  return res;
}

// refer to :http://www.cnblogs.com/singlex/p/pose_estimation_1_1.html
// 根据输入的参数将图像坐标转换到相机坐标中
// 输入为图像上的点坐标
// double distance 物距
// 输出3d点坐标的单位与distance（物距）的单位保持一致
cv::Point3f MonoMeasureTool::unproject(cv::Point2f p, double distance)
{
  auto fx = camera_intrinsic_.ptr<double>(0)[0];
  auto u0 = camera_intrinsic_.ptr<double>(0)[2];
  auto fy = camera_intrinsic_.ptr<double>(1)[1];
  auto v0 = camera_intrinsic_.ptr<double>(1)[2];

  double zc = distance;
  double xc = (p.x - u0) * distance / fx;
  double yc = (p.y - v0) * distance / fy;
  return cv::Point3f(xc, yc, zc);
}

// 获取image任意点的视角，pitch，yaw（相对相机坐标系）。
// 与相机坐标系保持一致。
void MonoMeasureTool::calcViewAngle(cv::Point2f p, float & pitch, float & yaw)
{
  auto fx = camera_intrinsic_.ptr<double>(0)[0];
  auto u0 = camera_intrinsic_.ptr<double>(0)[2];
  auto fy = camera_intrinsic_.ptr<double>(1)[1];
  auto v0 = camera_intrinsic_.ptr<double>(1)[2];

  pitch = atan2((p.y - v0), fy);
  yaw = atan2((p.x - u0), fx);
}

bool MonoMeasureTool::calcArmorTarget(
  const ArmorObject & obj, cv::Point3f & position, cv::Mat & rvec, std::string & armor_type)
{ 
  if(obj.is_ok)
  { 
    
    if (is_big_armor(obj)) {
      armor_type = "large";
      return solvePnp(obj.pts_binary, BIG_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
    } else {
      armor_type = "small";
      return solvePnp(obj.pts_binary, SMALL_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
    }
   

  }
  else {
  if (is_big_armor(obj)) {
    armor_type = "large";
    return solvePnp(obj.pts, BIG_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
  } else {
    armor_type = "small";
    return solvePnp(obj.pts, SMALL_ARMOR_3D_POINTS, position, rvec, cv::SOLVEPNP_IPPE);
 }
}
}

float MonoMeasureTool::calcDistanceToCenter(const ArmorObject & obj)
{
  cv::Point2f img_center(
    this->camera_intrinsic_.at<double>(0, 2), this->camera_intrinsic_.at<double>(1, 2));
  cv::Point2f armor_center;
  if(obj.is_ok)
  {
  armor_center.x = (obj.pts_binary[0].x + obj.pts_binary[1].x + obj.pts_binary[2].x + obj.pts_binary[3].x) / 4.;
  armor_center.y = (obj.pts_binary[0].y + obj.pts_binary[1].y + obj.pts_binary[2].y + obj.pts_binary[3].y) / 4.;
  }
  else {
  armor_center.x = (obj.pts[0].x + obj.pts[1].x + obj.pts[2].x + obj.pts[3].x) / 4.;
  armor_center.y = (obj.pts[0].y + obj.pts[1].y + obj.pts[2].y + obj.pts[3].y) / 4.;
}
  auto dis_vec = img_center - armor_center;
  return sqrt(dis_vec.dot(dis_vec));
}

}  // namespace rm_auto_aim
