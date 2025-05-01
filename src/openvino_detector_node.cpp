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

#include <cv_bridge/cv_bridge.h>
#include <fmt/format.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <iostream>
#include <ostream>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>

#include "armor_detector_openvino/detector_openvino_node.hpp"

namespace rm_auto_aim
{
ArmorDetectorOpenvinoNode::ArmorDetectorOpenvinoNode(const rclcpp::NodeOptions & options)
: Node("armor_detector_openvino", options)
{
  RCLCPP_INFO(
    this->get_logger(), "Initializing armor_detector_openvino with intra-process communication %s",
    options.use_intra_process_comms() ? "enabled" : "disabled");

  callback_group_img_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  detector_ = nullptr;
  this->initDetector();
  if (!detector_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenVINO");
    return;
  }
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  detect_color_ = this->declare_parameter("detect_color", 0, param_desc);

  auto use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);

  camera_name_ = this->declare_parameter("detector.camera_name", "front_industrial_camera");
  transport_type_ =
    this->declare_parameter("detector.subscribe_compressed", false) ? "compressed" : "raw";

  RCLCPP_INFO(
    this->get_logger(), "camera_name: %s, transport_type: %s", camera_name_.c_str(),
    transport_type_.c_str());

  max_infer_running_=this->declare_parameter("detector.max_infer_running", 5);
  RCLCPP_INFO(this->get_logger(), "max_infer_running: %d", max_infer_running_);

  // Debug mode handler
  RCLCPP_INFO(this->get_logger(), "Setup debug_mode handler");
  debug_mode_ = this->declare_parameter("debug_mode", true);
  if (debug_mode_) {
    this->createDebugPublishers();
  }
  // Register debug mode param handler
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("debug_mode", [this](const rclcpp::Parameter & p) {
      this->debug_mode_ = p.as_bool();
      this->debug_mode_ ? this->createDebugPublishers() : this->destroyDebugPublishers();
    });

  RCLCPP_INFO(this->get_logger(), "Setup ROS subs pubs");
  // Armors publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
    "detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detector/marker", 10);
  
  // Camera handler
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    
     camera_name_ +"/camera_info", use_sensor_data_qos ? rclcpp::SensorDataQoS() : rclcpp::QoS(1),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {

      this->cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      this->measure_tool_ = std::make_unique<MonoMeasureTool>(
        std::vector<double>(this->cam_info_->k.begin(), this->cam_info_->k.end()),
        this->cam_info_->d);

      RCLCPP_INFO(
        this->get_logger(), "%s",
        fmt::format(
          "Camera intrinsic: {} \ncamera distortion: {}", fmt::join(this->cam_info_->k, " "),
          fmt::join(this->cam_info_->d, " "))
          .c_str());

      // Release subscription
      this->cam_info_sub_.reset();      
    });

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = callback_group_img_;

 
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    camera_name_ + "/image",rclcpp::SensorDataQoS(),
    std::bind(&ArmorDetectorOpenvinoNode::imgCallback, this, std::placeholders::_1),sub_options
    );

  RCLCPP_INFO(this->get_logger(), "Initializing finished.");
  last_stat_time_ = this->now();
}
void ArmorDetectorOpenvinoNode::initDetector()
{
  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector_openvino");
  auto classify_model_path = pkg_path + "/model/mlp.onnx";
  auto classify_label_path = pkg_path + "/model/label.txt";
  auto model_path = this->declare_parameter("detector.model_path", "/home/hy/pb2025_sentry_ws/src/pb2025_rm_vision/armor_detector_openvino/model/opt-1208-001.onnx");
  auto device_type = this->declare_parameter("detector.device_type", "GPU");
  float conf_threshold = this->declare_parameter("detector.confidence_threshold", 0.25);
  int top_k = this->declare_parameter("detector.top_k", 128);
  float nms_threshold = this->declare_parameter("detector.nms_threshold", 0.3);

  float expand_ratio_w = this->declare_parameter("light.expand_ratio_w", 2.0);
  float expand_ratio_h = this->declare_parameter("light.expand_ratio_h", 1.5);
  int binary_thres = this->declare_parameter("light.binary_threshold", 85);

  LightParams l_params = {
    .min_ratio = declare_parameter("light.min_ratio", 0.08),
    .max_ratio = declare_parameter("light.max_ratio", 0.4),
    .max_angle = declare_parameter("light.max_angle", 40.0),
    };

  if (model_path.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Require model_path but got empty.");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Model path: %s", model_path.c_str());

  // Create detector
  detector_ = std::make_unique<DetectorOpenVino>(
    model_path, classify_model_path, classify_label_path, device_type,l_params, conf_threshold, top_k,
    nms_threshold, expand_ratio_w, expand_ratio_h, binary_thres);

  detector_->detect_color_=detect_color_;
  // Set detect callback
  detector_->setCallback(std::bind(
    &ArmorDetectorOpenvinoNode::openvinoDetectCallback, this, std::placeholders::_1,
    std::placeholders::_2, std::placeholders::_3));
  // init detector
  detector_->init();

}

void ArmorDetectorOpenvinoNode::printStats()
{
  using namespace std::chrono;
  
  auto now = steady_clock::now();
  
  if (last_stat_time_steady_.time_since_epoch().count() == 0) {
    last_stat_time_steady_ = now;
    return;
  }

  auto elapsed = duration_cast<duration<double>>(now - last_stat_time_steady_);
  if (elapsed.count() >= 1.0) {
    RCLCPP_INFO(this->get_logger(),
      "Image recv: %zu Hz, Detect finished: %zu Hz, Inference queue: %d",
      img_recv_count_, detect_finish_count_, infer_running_count_.load()
    );

    img_recv_count_ = 0;
    detect_finish_count_ = 0;
    last_stat_time_steady_ = now;
  }
}



void ArmorDetectorOpenvinoNode::imgCallback(const sensor_msgs::msg::Image::ConstSharedPtr  msg)
{  img_recv_count_++;


  if (infer_running_count_.load() >= max_infer_running_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
      "Infer running too much (%d), dropping frame", infer_running_count_.load());
      return;    
  }
  
  auto timestamp = rclcpp::Time(msg->header.stamp);
  auto now = this->now();
  //int64_t latency_ns = now.nanoseconds() - timestamp.nanoseconds();
 // RCLCPP_INFO(this->get_logger(), "Latency: %.1f ms", latency_ns / 1e6);
  frame_id_ = msg->header.frame_id;
  auto img = cv_bridge::toCvCopy(msg, "rgb8")->image;

  // push image to detector
  detector_->pushInput(img, timestamp.nanoseconds());
  auto end=this->now();
  infer_running_count_++;
  if(debug_mode_){
  printStats();}
  
}



void ArmorDetectorOpenvinoNode::openvinoDetectCallback(
  const std::vector<ArmorObject> & objs, int64_t timestamp_nanosec, const cv::Mat & src_img)
{ std::lock_guard<std::mutex> lock(callback_mutex_);
  detect_finish_count_++;
  infer_running_count_--;
  
  if (measure_tool_ == nullptr) {
    RCLCPP_WARN(this->get_logger(), "No camera_info receive yet.");
    return;
  }

  detect_color_ = this->get_parameter("detect_color").as_int();

  auto timestamp = rclcpp::Time(timestamp_nanosec);

  // Used to draw debug info
  cv::Mat debug_img;
  if (debug_mode_) {
    debug_img = src_img.clone();
  }

  auto_aim_interfaces::msg::Armors armors_msg;
  armors_msg.header.frame_id = frame_id_;
  armors_msg.header.stamp = timestamp;
  armor_marker_.header = text_marker_.header = armors_msg.header;

  armors_msg.armors.clear();
  marker_array_.markers.clear();
  armor_marker_.id = 0;
  text_marker_.id = 0;

  for (auto & obj : objs) {
    if (detect_color_ == 0 && obj.color != ArmorColor::RED) {
      continue;
    } else if (detect_color_ == 1 && obj.color != ArmorColor::BLUE) {
      continue;
    }

    auto_aim_interfaces::msg::Armor armor;

    cv::Point3f target_position;
    cv::Mat target_rvec;
    std::string armor_type;

    if (!measure_tool_->calcArmorTarget(obj, target_position, target_rvec, armor_type)) {
      RCLCPP_WARN(this->get_logger(), "Calc target failed.");
      continue;
    }
    debug_number_pub_.publish(cv_bridge::CvImage(armors_msg.header, "mono8", obj.number_img).toImageMsg());
    // if(obj.confidence ==0)
    //   continue;
    

    cv::Mat rot_mat;
    cv::Rodrigues(target_rvec, rot_mat);
    tf2::Matrix3x3 tf_rot_mat(
      rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),
      rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),
      rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
    tf2::Quaternion tf_quaternion;
    tf_rot_mat.getRotation(tf_quaternion);
    

    armor.number = K_ARMOR_NAMES[static_cast<int>(obj.number)];
    armor.type = armor_type;
    armor.pose.position.x = target_position.x;
    armor.pose.position.y = target_position.y;
    armor.pose.position.z = target_position.z;
    armor.pose.orientation.x = tf_quaternion.x();
    armor.pose.orientation.y = tf_quaternion.y();
    armor.pose.orientation.z = tf_quaternion.z();
    armor.pose.orientation.w = tf_quaternion.w();
    armor.distance_to_image_center = measure_tool_->calcDistanceToCenter(obj);

    // Fill the markers
    armor_marker_.id++;
    armor_marker_.scale.y = armor.type == "small" ? 0.135 : 0.23;
    armor_marker_.pose = armor.pose;

    text_marker_.id++;
    text_marker_.pose.position = armor.pose.position;
    text_marker_.pose.position.y -= 0.1;
    text_marker_.text = armor.number;

    armors_msg.armors.emplace_back(armor);
    marker_array_.markers.emplace_back(armor_marker_);
    marker_array_.markers.emplace_back(text_marker_);

    publishMarkers(objs.empty());

    

    if (debug_mode_) {
      if (debug_img.empty()) {
        // Avoid debug_mode change in processing
        continue;
      }

      // Draw armor
   
      static const int next_indices[] = {2, 0, 3, 1};

    for (size_t i = 0; i < 4; ++i) {
        cv::line(debug_img, obj.pts[i], obj.pts[(i + 1) % 4], cv::Scalar(255, 48, 48), 1);
        if (obj.is_ok) {
            cv::line(debug_img, obj.pts_binary[i], obj.pts_binary[next_indices[i]], cv::Scalar(0, 255, 0), 1);
        }    
    }


      std::string armor_color;
      switch (obj.color) {
        case ArmorColor::BLUE:
          armor_color = "B";
          break;
        case ArmorColor::RED:
          armor_color = "R";
          break;
        case ArmorColor::NONE:
          armor_color = "N";
          break;
        case ArmorColor::PURPLE:
          armor_color = "P";
          break;
        default:
          armor_color = "UNKNOWN";
          break;
      }

      std::string armor_key = fmt::format("{} {}", armor_color, static_cast<int>(obj.number));
      cv::putText(
        debug_img, armor_key, cv::Point2i(obj.pts[0]), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 255), 2);
    }
  }

  armors_pub_->publish(armors_msg);
  
  if (debug_mode_) {
    if (debug_img.empty()) {
      // Avoid debug_mode change in processing
      return;
    }

    cv::circle(
      debug_img, cv::Point2i(cam_info_->width / 2., cam_info_->height / 2.), 5,
      cv::Scalar(255, 0, 0), 2);

    auto end = this->now();
    auto duration = end.seconds() - timestamp.seconds();
    std::string latency = fmt::format("Latency: {:.3f}ms", duration * 1000);
    

    cv::putText(
      debug_img, latency, cv::Point2i(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);

    debug_img_pub_.publish(cv_bridge::CvImage(armors_msg.header, "rgb8", debug_img).toImageMsg());
    
  }
  
}

void ArmorDetectorOpenvinoNode::publishMarkers(bool is_empty)
{
  using Marker = visualization_msgs::msg::Marker;

  if (is_empty) {
    visualization_msgs::msg::MarkerArray delete_markers;

    Marker armor_delete_marker;
    armor_delete_marker.action = Marker::DELETEALL;
    armor_delete_marker.ns = "armors";
    armor_delete_marker.header.stamp = this->now();
    delete_markers.markers.push_back(armor_delete_marker);

    Marker text_delete_marker;
    text_delete_marker.action = Marker::DELETEALL;
    text_delete_marker.ns = "classification";
    text_delete_marker.header.stamp = this->now();
    delete_markers.markers.push_back(text_delete_marker);
  } else {
    marker_pub_->publish(marker_array_);
  }
  
}

void ArmorDetectorOpenvinoNode::createDebugPublishers()
{
  debug_img_pub_ = image_transport::create_publisher(this, "detector/debug_img");
  debug_number_pub_ = image_transport::create_publisher(this,"detector/debug_number");
}

void ArmorDetectorOpenvinoNode::destroyDebugPublishers() { debug_img_pub_.shutdown(); }

}  // namespace rm_auto_aim

// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorOpenvinoNode)



int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rm_auto_aim::ArmorDetectorOpenvinoNode>(rclcpp::NodeOptions());

  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 
    std::thread::hardware_concurrency()  
  );

  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
