/*#include <SpinnakerCamera/Camera.h>
#include <LightStage/LightStage.hh>
#include <SpinnakerCamera/CameraUtils.h>
#include <SpinnakerCamera/CaptureConfig.h>
#include <SpinnakerCamera/Decoder.h>
#include <SpinnakerCamera/MeanProcessor.h>
#include <SpinnakerCamera/MedianProcessor.h>
#include <SpinnakerCamera/sinks/JpegWriter.h>
#include <SpinnakerDefs.h>
#include <cstddef>*/

#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <half.h>
#include <light_stage_test/Vector.hh>
#include <math.h>
//#include <corecrt_math.h>
#include <ImathVec.h>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <chrono>



#include <SpinnakerCamera/sinks/ExrWriter.h>

#include "merl_base_brdf.h"
//#include <winnls.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace std::chrono;

// constants

float k_M_PI = static_cast<float>(M_PI);

//============== HELPER FUNCTIONS ===================================

// point on circle to point on sphere -> only z has to be calculated
float get_z(int x, int y, int radius, Vec3f centerpoint) {

  return static_cast<float>(sqrt(pow(radius, 2) - pow(x, 2) - pow(y, 2))) +
         centerpoint[2];
}

// calculate Angle between 2 vectors
// TODO: not norm use sum instead
float getAngleBetweenVectors(Vec3f first, Vec3f second) {
  return acos(first.dot(second)) /
         ((abs(first[0]) + abs(first[1]) + abs(first[2])) *
          (abs(second[0]) + abs(second[1]) + abs(second[2])));
}

float calcFL(String path, float real_world_object_size,
             float real_world_object_distance) {

  Mat img = imread(path, IMREAD_COLOR);
  vector<Vec3f> circ;
  // Convert to gray-scale
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);

  // Blur the image to reduce noise
  Mat img_blur;
  medianBlur(gray, img_blur, 5);

  // Apply Hough Transform
  HoughCircles(img_blur, circ, HOUGH_GRADIENT, 1, img_blur.rows / 2, 120, 30,
               300, 1600); // Canny 191 for test image

  float diameter = circ[0][2];

  return diameter * real_world_object_distance / real_world_object_size;
}

bool almostequal(float num1, float num2, int ulp) {
  return std::fabs(num1 - num2) <= std::numeric_limits<float>::epsilon() *
                                       std::fabs(num1 + num2) * ulp
         // unless the result is subnormal
         || std::fabs(num1 - num2) < std::numeric_limits<float>::min();
}

Vec3f getbinormalvector(Vec3f normalVector, Vec3f pointonsphere) {

  // getting arbitrary point on the tangent plane with the coords x unknown,
  // pointonsphere.y, pointonsphere.z + 1 float x = (normalVector[0] *
  // pointonsphere[0] -
  //           normalVector[1] * pointonsphere[1] +
  //           normalVector[1] * pointonsphere[1] -
  //           normalVector[2] *(pointonsphere[2] + 1)
  //         + normalVector[2] * pointonsphere[2]) / normalVector[0];

  float z =
      (1.0f / normalVector[2]) * ((-1.0f) * normalVector[0] * pointonsphere[0] -
                                  normalVector[1] * pointonsphere[1]);

  Vec3f tangentvec(pointonsphere[0], pointonsphere[1], z);

  // the binormal should be orthogonal to the normal vector

  if (almostequal(tangentvec.dot(normalVector), 0.0f, 2)) {

    return normalize(tangentvec);
  } else {
    // std::cout << std::fixed << std::setprecision(20)  << "dot tangentvec and
    // normal = " << std::to_string(tangentvec.dot(normalVector)) << " \n";
    return Vec3f(0.f, 0.f, 0.f);
  }
}

float calcfov(float img_dim, float focal_length) {
  return atan(img_dim / (focal_length * 2.f)) * 2.f;
}

// calculate normalized device coords (NDC)
float calcpxNDC(float pixel, float imagedim) {
  // we want the final camera ray to pass through the middle of the pixel thats
  // why 0.5 is added
  return (pixel + 0.5f) / imagedim;
}

Vec3f generateRay(Point p, float roi_offset_x, float roi_offset_y,
                  float roi_width, float roi_height, float img_height,
                  float img_width, float fovx, float fovy) {

  // project the point from the crop out coords to the whole Image coords
  // Point2f p_projected(roi_offset_x + p.x, roi_offset_y + p.y);

  // float imageAspectRatio = img_height / img_width; // assuming width > height

  float NDCx =
      calcpxNDC(static_cast<float>(p.x), img_width); //* imageAspectRatio;
  float NDCy = calcpxNDC(static_cast<float>(p.y), img_height);
  // cout << "NDCx, NDCy is " << to_string(NDCx) << ", " << to_string(NDCy) <<
  // "\n";

  float px = ((2.f * NDCx) - 1.0f) * tan(fovx / 2.f);
  float py = (1.f - (2.f * NDCy)) * tan(fovy / 2.f);

  // cout << "px, py is " << to_string(px) << ", " << to_string(py) << "\n";
  Vec3f rayOrigin(0, 0, 0);
  Vec3f rayDirection = Vec3f(px, py, 1) - rayOrigin;

  return normalize(rayDirection);
}

float ray_sphere_intersec(Vec3f ray_norm_d, Vec3f ray_origin,
                          Vec3f sphere_origin, float sphererad) {
  // ray sphere intersection spits out t1,t2 both are hitpoints on the sphere ,
  // if decides which is the right ones

  Vec3f o_minus_c = ray_origin - sphere_origin;

  float p = ray_norm_d.dot(o_minus_c);
  float q = o_minus_c.dot(o_minus_c) - pow(sphererad, 2.f);
  float discriminant = (p * p) - q;
  // float p = 2 * ray_norm_d.ddot(ray_origin + sphere_origin);
  // float q = sphere_origin.ddot(sphere_origin) + ray_origin.ddot(ray_origin)
  // - 2 * sphere_origin.ddot(ray_origin) - sphererad*sphererad;

  // float discriminant = (0.5*p)*(0.5*p) - q;

  if (discriminant < 0.f) {
    return 0;
  }
  float dRoot = sqrt(discriminant);
  //  float t1 = -0.5*p + dRoot;
  //  float t2 = -0.5*p - dRoot;
  float t1 = -p - dRoot;
  float t2 = -p + dRoot;

  if (t1 < t2) {
    return t1;
  } else {
    return t2;
  }
}

float getdistance2d(Point2f a, Point2f b) {

  return sqrt((pow(a.x - b.x, 2.f)) + pow(a.y - b.y, 2.f));
}

float getdistance3d(Vec3f a, Vec3f b) {
  return sqrt(pow(a[0] - b[0], 2.f) + pow(a[1] - b[1], 2.f) +
              pow(a[2] - b[2], 2.f));
}

//============== MAIN ===================================

int main() {

  float distance_cam_spherecap = 0;
  float real_world_size_sphere = 0.1f;//0.064f;  // Size of sphere 100mm ~ 10cm ~ 0.1 m,
  float real_world_size_sphere_radius = real_world_size_sphere / 2.f;
  float FL = 10248;
  Point offset(500, 1300);
  float px_in_m = 3.5e-6f;

  // WorldSpace Setup
  Vec3f CameraOrigin(0, 0, 0);
  Vec3f forward(0, 0, 1);
  Vec3f right(1, 0, 0);
  Vec3f up(0, -1, 0);

  float fovx = calcfov(4096.0, FL); // FOV for Width of Image
  float fovy = calcfov(3000.0, FL); // FOV for Height of image

  // SET FILENAMES HERE
  String folder = "greychristmas/";

  // DIFFERENT HEMISPHERES ON THE LS

  // static const unsigned char new_hemi[] = {
  //     112, 113, 111, 98,  109, 87,  86,  83,  70,  56,  57,  58,  99,
  //     109, 114, 115, 108, 107, 100, 90,  91,  84,  82,  71,  69,  59,
  //     73,  68,  60,  61,  64,  66,  67,  65,  63,  62,  97,  77,  76,
  //     75,  74,  73,  81,  80,  79,  79,  78,  94,  96,  95,  93,  92,
  //     89,  90,  88,  101, 100, 107, 108, 110, 106, 116, 123, 119, 117,
  //     118, 127, 126, 125, 124, 120, 122, 121, 105, 102, 104, 103, 72 ,

  // 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
  // 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
  // 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 };

  static const unsigned char main_arc[] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37,
      38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

  static const unsigned char upper_hemi[] = {
      // 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
      // 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,
      // 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
      // 47,  48,  49,  50,  51,  52,  53,  54,
      
      59,  60,  61,  62,  63,  64,  65,
      66,  67,  68,  69,  73,  74,  75,  76,  77,  78,  79,  80,  89,  92,  93,
      94,  95,  96,  97,  101, 102, 103, 104, 105, 106, 110, 115, 116, 117, 118,
      119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
      134, 135, 145, 146, 147, 148, 149, 150, 151, 152, 153, 156, 157, 158, 159,
      160, 161, 162, 163, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 176,
      177, 178, 179, 180, 181, 185, 187, 188, 189, 190, 191, 192, 195

   };

  static const unsigned char test_pics[] = {20};

  // iterates through all lights
  // Init Circle
  // Create a vector for detected circles
  vector<Vec3f> circles;

  // initialize the BRDF
  MerlBaseBrdf test;

  auto start = high_resolution_clock::now();


  /*open up stream to write out point to plot them
      std::fstream bfile =
     std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/points_.binary",
     std::ios::out | std::ios::binary); std::fstream wo =
     std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/wo_.binary",
     std::ios::out | std::ios::binary); std::fstream wI =
     std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/wi_.binary",
     std::ios::out | std::ios::binary); std::fstream locations =
     std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/locations_.binary",
     std::ios::out | std::ios::binary);
  */

  // create circle Mask

  Mat img_mask = imread("/home/stratman/Documents/code_public/"
                        "light_stage_test/HDR_pics/" +
                            folder + "hdr_image_final_22.exr",
                        IMREAD_COLOR);

  // Convert to gray-scale
  Mat gray;
  cvtColor(img_mask, gray, COLOR_BGR2GRAY);

  // Blur the image to reduce noise
  Mat img_blur;
  medianBlur(gray, img_blur, 3);

  // =================================Apply Hough Transform ==========================
  HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img_blur.rows / 4, 0.98,
               0.98, 300, 1200); 

  if (circles.empty()) {
    std::cout << "circles is empty! no circle detected - change parameters \n";
  }

  Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
  int radius = cvRound(circles[0][2]) + 5;

  cout << "Circle successfully detected ! \n";
  // preliminary setup for calculating the right and left circle points in 3d

  center = Point(center.x + offset.x, center.y + offset.y);
  Point leftcirc = Point(center.x - radius, center.y);
  Point rightcirc = Point(center.x + radius - 1, center.y);

  // with radius calculate the real world to pixel
  // https://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image
  float FL_in_m = FL / px_in_m;
  float sphere_size_in_m = (static_cast<float>(radius) * 2.f) / px_in_m;
  distance_cam_spherecap = real_world_size_sphere * FL_in_m / sphere_size_in_m;
  float fov_real =
      calcfov(real_world_size_sphere, distance_cam_spherecap) / 2.f;
  float rayl_dist_cam = distance_cam_spherecap / cos(fov_real);

  // calculate the direction vectors , Points in Opencv are y,x
  Vec3f dir_a = generateRay(leftcirc, offset.y, offset.x, 1400.0, 1400.0,
                            3000.0, 4096.0, fovx, fovy);
  Vec3f dir_b = generateRay(rightcirc, offset.y, offset.x, 1400.0, 1400.0,
                            3000.0, 4096.0, fovx, fovy);

  Vec3f ray_left_circ = CameraOrigin + rayl_dist_cam * dir_a;
  Vec3f ray_right_circ = CameraOrigin + rayl_dist_cam * dir_b;

  float distancebetween_l_r_rays =
      (getdistance3d(ray_left_circ, ray_right_circ) / 2.f);

  // triangle calculations to get the radius of the given sphere
  // https://www.blocklayer.com/trig/right-triangles
  float theta = asin(distancebetween_l_r_rays / distance_cam_spherecap);
  cout << "Distance between enclosing rays: " << distancebetween_l_r_rays
       << ", distance_cam_cap_: " << distance_cam_spherecap
       << ", radius: " << radius << "\n";
  // distance between enclosing rays (/2): 0.0317374 ,
  // distance_cam_cap_: 1.05446radius: 311 diff_rad_distance: 9.67354e-05

  float beta = k_M_PI - (theta + (k_M_PI / 2.f));
  float beta_ = (k_M_PI / 2.f) - beta;

  cout << "theta: " << theta << "beta_: " << beta_ << "\n";

  // cos(beta_) = a / r
  // r = a / cos(beta_)
  float sphere_radius = distancebetween_l_r_rays / cos(beta_);
  float diff_rad_distance = sin(beta_) * static_cast<float>(sphere_radius);
  float distance_cam_sphere = distance_cam_spherecap + diff_rad_distance;
  cout << "sphere_radius: " << sphere_radius << "\n"
       << "diff_rad_distance: " << diff_rad_distance << "\n"
       << ", distance_cam_sphere: " << distance_cam_sphere << "\n";

  // ALTERNATIVE
  float diff_rad_alternative = sin(theta) * real_world_size_sphere_radius;
  float distance_between_rays_alternative =
      cos(theta) * real_world_size_sphere_radius;
  float distance_between_spherecap_cam =
      real_world_size_sphere_radius / tan(theta);

  float distance_cam_sphere_alternative =
      distance_between_spherecap_cam + diff_rad_alternative;
  cout << "=======ALTERNATIVE ====== \n";
  cout << "Distance between enclosing rays: "
       << distance_between_rays_alternative << "\n"
       << ", distance_between_spherecap_cam: " << distance_between_spherecap_cam
       << "\n"
       << ", diff_rad_alternative: " << diff_rad_alternative << "\n"
       << ", distance_cam_sphere_alternative: "
       << distance_cam_sphere_alternative << "\n";

  sphere_radius = real_world_size_sphere_radius;

  distance_cam_sphere = distance_cam_sphere_alternative;

  Vec3f CenterPoint_dir = generateRay(center, 1300.0, 500.0, 1400.0, 1400.0,
                                      3000.0, 4096.0, fovx, fovy);

  Vec3f CenterPoint = CameraOrigin + CenterPoint_dir * distance_cam_sphere;

cout << "Center Point is: " << CenterPoint_dir << "\n";

  // Vec3f ray_r_dir = generateRay(rightcirc, 1300.0, 500.0, 1400.0, 1400.0,
  //                               3000.0, 4096.0, fovx, fovy);
  // float r_length = ray_sphere_intersec(ray_r_dir, CameraOrigin, CenterPoint,
  //                                      real_world_size_sphere_radius);

  // Vec3f ray_l_dir = generateRay(leftcirc, 1300.0, 500.0, 1400.0, 1400.0,
  // 3000.0,
  //                               4096.0, fovx, fovy);
  // float l_length = ray_sphere_intersec(ray_l_dir, CameraOrigin, CenterPoint,
  //                                      real_world_size_sphere_radius);

  cout << "Center Point calculated, starting to loop through pics now ! \n";

  //================= MAIN LOOP TO ITERATE THROUGH PICTURES
  //===============================

  std::fstream locations =
      std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                   "visualizebrdfs/locations.binary",
                   std::ios::out | std::ios::binary);


     std::fstream wi_angles_bins =
        std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                     "visualizebrdfs/wi_angles.binary",
                     std::ios::out | std::ios::binary);      

  //======TESTING FUNCTIONS HERE ======
  assert(5 ==
         ray_sphere_intersec(Vec3f(0, 0, 1), CameraOrigin, Vec3f(0, 0, 6), 1));
  assert(0 ==
         ray_sphere_intersec(Vec3f(0, 1, 0), CameraOrigin, Vec3f(0, 0, 6), 1));

  //#pragma omp parallel for
  for (unsigned char elem : upper_hemi) {

    // set filenames here
    String pic_name = folder + "hdr_image_final_" + std::to_string(elem);
    String pic_name_raw = folder + "hdr_image_" + std::to_string(elem);

    //  Read the image as gray-scale
    Mat img = imread(
        "/home/stratman/Documents/code_public/light_stage_test/HDR_pics/" +
            pic_name + ".exr",
        IMREAD_UNCHANGED);

    // std::cout << img.at<Vec3f>(300, 700) << std::endl;
    // imshow("Test", img);
    // waitKey(0);
    Mat img_raw = imread(
        "/home/stratman/Documents/code_public/light_stage_test/HDR_pics/" +
            pic_name_raw + ".exr",
        IMREAD_UNCHANGED);
    // imshow("Test1", img_raw);
    // waitKey(0);

    // Masks the region of the circle ------ DOESNT WORK RN ------
    // Mat circ_mask(img.size(), CV_8U, Scalar(0,0,0));
    // circle(circ_mask, center, radius, cv::Scalar(1) , -1, 8, 0);

    // // std::cout << img.size() << " " << circ_mask.size() << std::endl;
    // //imshow("beforemult", img);
    // //img = img.mul(circ_mask);
    // Mat temp(img.size(), img.type());
    // img.copyTo(temp, circ_mask);
    // img = temp;
    // imshow("circmask", circ_mask);
    // imshow("aftermult", img);
    // waitKey(0);

    // std::cout << img.at<Vec3f>(300, 700) << std::endl;

    // ========= Calculation of the Normal of the brightest spot on
    // the sphere ==========
    double max_val;
    Point max_loc;

    // Img is first blurred for better detection
    GaussianBlur(img_raw, img_raw, Size(5, 5), 3.0);

    // get max Location of the sphere
    cv::minMaxLoc(img_raw, NULL, &max_val, NULL, &max_loc);
    // max_loc = Point(max_loc.x + offset.x, max_loc.y + offset.y);

    // convert max_loc_3d to 3d
    // calculate normal of the point max_loc_3d to get the position of the light
    // ; distance is at camera distance

    Vec3f max_loc_3d_dir =
        generateRay(Point(max_loc.x + offset.x, max_loc.y + offset.y), 1300.0,
                    500.0, 1400.0, 1400.0, 3000.0, 4096.0, fovx, fovy);

    Vec3f max_loc_3d =
        CameraOrigin +
        max_loc_3d_dir * ray_sphere_intersec(max_loc_3d_dir, CameraOrigin,
                                             CenterPoint, sphere_radius);

    Vec3f max_loc_normal = normalize(max_loc_3d - CenterPoint);

    Vec3f reflection_vector =
        normalize((max_loc_3d_dir - (2 * (max_loc_3d_dir.dot(max_loc_normal)) *
                                     max_loc_normal)));

    Vec3f lightlocation =
        max_loc_3d + reflection_vector * distance_cam_sphere_alternative;

    // Point2f principlepx = {2021.360907134241, 1477.123308582499};

    // this should test for true
    // float test1 = get_z(left_circ[0], left_circ[1], distance_cam_sphere,
    // CenterPoint); assert(left_circ[2] == test1 );

    //   Mat mask_raycast(img.size(), img.type());
    //   std::fstream a_dir_file =
    //   std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/a_dir_"
    //   + to_string(elem) + ".binary", std::ios::out | std::ios::binary);

    //     for (int i = 0; i < img.rows; ++i) {
    //           for (int j = 0; j < img.cols; ++j) {

    //     Point a(i + offset.x ,j + offset.y);
    //     Vec3f a_dir = generateRay(a, 1300.0f, 500.0f, 1400.0f, 1400.0f,
    //     3000.0f, 4096.0f, fovx, fovy);

    //     if(i % 10 == 0 && j % 10 == 0){
    //     a_dir_file.write((char*)&CameraOrigin, sizeof(cv::Vec3f));
    //     a_dir_file.write((char*)&a_dir, sizeof(cv::Vec3f));
    // }
    //     if (0.0 != ray_sphere_intersec(a_dir, CameraOrigin, CenterPoint,
    //     sphere_radius)){ cout<< "Pixel is colored" << "\n";
    //     mask_raycast.at<Vec3f>(Point(i,j)) = Vec3f(1.0f, 0 , 0);
    //     }
    //           }

    //           }
    //     //imshow("circmask", circ_mask);
    //     imshow("raymask", mask_raycast);
    //     //mask_raycast = circ_mask - mask_raycast;
    //     //imshow("raymask_multiplied", mask_raycast);
    //     waitKey(0);
    // //   a_dir_file.close();

    // open up stream to write out point to plot them
    std::fstream bfile =
        std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                     "visualizebrdfs/points_" +
                         to_string(elem) + ".binary",
                     std::ios::out | std::ios::binary);




    // std::fstream wo =
    // std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/wo_"
    // + to_string(elem) + ".binary", std::ios::out | std::ios::binary);
    // std::fstream wI =
    // std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/wi_"
    // + to_string(elem) + ".binary", std::ios::out | std::ios::binary);

    // locations.write((char*)&lightlocation, sizeof(cv::Vec3f));
    int idx = 0;
    Mat brdfdebug_angles(img.size(), CV_32FC3, 0.0);
    Mat wi_angles(img.size(), CV_32FC3, 0.0);
    Mat w0_angles(img.size(), CV_32FC3, 0.0);
    Mat normals(img.size(), CV_32FC3, 0.0);

    // calculate all angles and image intensities for the brdf

    // brdfdebug_angles.at<Vec3f>(rightcirc.y - offset.y, rightcirc.x -
    // offset.x) = Vec3f(0,0,1); brdfdebug_angles.at<Vec3f>(leftcirc.y -
    // offset.y ,leftcirc.x - offset.x) = Vec3f(0,0,1);

    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {

        // check if the checked pixels lies inside the sphere, if not continue
        Point arb(j + offset.x, i + offset.y);
        // if (getdistance2d(arb, center) > radius) {
        //  continue;
        // }

        // Matrix elements can be accessed via : I.at<uchar>(i,j)
        cv::Vec3f intensity = img.at<cv::Vec3f>(i, j);

        // if (intensity[0] > 800.f && intensity[1] > 800.f && intensity[2] >
        // 800.f ) {

        //   cout<< "stop\n";
        // }

        // Point of arbitrary point on the sphere is calcualted here
        // first the direction is generated with the help of the priorly used
        // generateRay function
        Vec3f arbi_3DPoint_dir = generateRay(arb, 1300.0, 500.0, 1400.0, 1400.0,
                                             3000.0, 4096.0, fovx, fovy);

        // then, with the distance calculated from the ray-sphere trace we get
        // the vector arbi_3DPoint
        Vec3f arbi_3DPoint =
            CameraOrigin +
            arbi_3DPoint_dir * ray_sphere_intersec(arbi_3DPoint_dir,
                                                   CameraOrigin, CenterPoint,
                                                   sphere_radius);

        // if the 3dPoint is at 0,0,0 the distance of the ray sphere intersect
        // is 0, which means the ray-sphere-intersect returns 0 as in it doesnt
        // hit the sphere should result in the similar
        if (arbi_3DPoint == Vec3f(0.f, 0.f, 0.f)) {
          // cout << "arbi3dPoint is 0 ,0 ,0 !!! \n";
          continue;
        }

        // setup wi, wo and normal of the arbitrary point
        Vec3f arbi_3DPoint_normal = normalize(
            arbi_3DPoint - CenterPoint); // normal of the arbitrary point
        Vec3f wi = lightlocation - arbi_3DPoint;      // ray towards the light
        intensity = intensity * 1 / pow(norm(wi), 2); // foreshortening
        wi = normalize(wi);
        Vec3f w0 = normalize(CameraOrigin -
                             arbi_3DPoint); // ray towards the Camera Origin

        float angle_wi_normal = getAngleBetweenVectors(wi, arbi_3DPoint_normal);
        float angle_w0_normal = getAngleBetweenVectors(w0, arbi_3DPoint_normal);

        // float anglewowi = getAngleBetweenVectors(wi, w0);

        // if(angle_wi_normal > 0.08f){
        // brdfdebug_angles.at<Vec3f>(i, j) =  Vec3f(angle_wi_normal , 0, 0);
        // }

        Vec3f half_vec = normalize((wi + w0));
        float angle_wi_h = getAngleBetweenVectors(wi, half_vec);

        brdfdebug_angles.at<Vec3f>(i, j) =
            Vec3f(std::clamp(half_vec.dot(wi), 0.f, 1.f), angle_w0_normal,
                  angle_wi_normal);
        wi_angles.at<Vec3f>(i, j) = wi;
        w0_angles.at<Vec3f>(i, j) = w0;
        normals.at<Vec3f>(i, j) = arbi_3DPoint_normal;

        wi_angles_bins.write((char *)&angle_wi_normal, sizeof(float));

        // if angle between wi & normal or wo
        //& normal greater than 90 degree (Pi /2) -> throw them out
        if ((angle_wi_normal > (k_M_PI / 2.f)) ||
            (angle_w0_normal > (k_M_PI / 2.f))) {

          // << " and a wo angle of: " << to_string(angle_w0_normal) << "\n";

          continue;
        }
        //wi_angles_bins.write((char *)&angle_w0_normal, sizeof(float));
        wi_angles_bins.write((char *)&angle_wi_normal, sizeof(float));

        //===== Calculating bitangent and tangent =====
        base::Vec3f normal(arbi_3DPoint_normal[0], arbi_3DPoint_normal[1],
                           arbi_3DPoint_normal[2]);
        base::Vec3f bitangent, tangent;
        // if the angle between the normal and the center point direction is
        // smaller than ~5 degree, use another method to determine the tangents
        // and bitangents if ((getAngleBetweenVectors(arbi_3DPoint_normal,
        // Vec3f(-CenterPoint_dir[0] , -CenterPoint_dir[1],
        // -CenterPoint_dir[2])) < 0.08)) { Vec3f binormal =
        // getbinormalvector(arbi_3DPoint_normal, arbi_3DPoint);

        // tangent = base::Vec3f(binormal[0], binormal[1], binormal[2]);
        // bitangent = base::cross(tangent,base::Vec3f(arbi_3DPoint_normal[0],
        //                                       arbi_3DPoint_normal[1],
        //                                       arbi_3DPoint_normal[2]));

        // if(bitangent[0] == 0) {

        //   cout << "bitangent = 0 \n";
        // }

        // } else{
        // calculate tangent and bitangent of our brdf
        // between the ray that goes along the center point vector (from pole to
        // pole) and a orthogonal ray from the dissection ray to the
        // arbi_3DPoint the angles between those is called ang
        float ang =
            getAngleBetweenVectors(CenterPoint, (arbi_3DPoint - CenterPoint));
        float dist = getdistance3d(CenterPoint, arbi_3DPoint);
        float dist_ortho_vec = dist * cos(ang);
        Vec3f temptopoint =
            normalize(arbi_3DPoint -
                      (CameraOrigin + (distance_cam_sphere - dist_ortho_vec) *
                                          CenterPoint_dir));

        base::Vec3f topoint =
            base::Vec3f(temptopoint[0], temptopoint[1], temptopoint[2]);

        tangent = (base::cross(topoint, base::Vec3f(CenterPoint_dir[0],
                                                    CenterPoint_dir[1],
                                                    CenterPoint_dir[2])))
                      .normalize();
        bitangent = (base::cross(tangent, normal)).normalize();

     

        // recompute tangent to ensure orthonomality
        tangent = (base::cross(normal, bitangent)).normalize();
        // }

        // float theta_i = getAngleBetweenVectors(lightlocation,
        // px_loc_3d_normal); float theta_r =
        // getAngleBetweenVectors(sphere_center, px_loc_3d_normal);

        if (j % 100 == 0 && i % 100 == 0) {
          bfile.write((char *)&arbi_3DPoint, sizeof(cv::Vec3f));
          bfile.write((char *)&normal, sizeof(cv::Vec3f));
          bfile.write((char *)&tangent, sizeof(cv::Vec3f));
          bfile.write((char *)&bitangent, sizeof(cv::Vec3f));
          bfile.write((char *)&w0, sizeof(cv::Vec3f));
          bfile.write((char *)&wi, sizeof(cv::Vec3f));
        }

        idx++;
        // checking for nan
        assert(!(isnan(tangent[0])));
        assert(!(isnan(bitangent[0])));
        assert(!(isnan(wi[0])));
        assert(!(isnan(w0[0])));
        assert(!(isnan(intensity[0])));


        //               arbi_3DPoint_normal[2]) , tangent).normalize();
        // bitangent = base::cross(base::Vec3f(arbi_3DPoint_normal[0],
        // arbi_3DPoint_normal[1],
        //               arbi_3DPoint_normal[2]) , tangent).normalize();

        //    if ((i == 689 && j == 690) && elem == 20) {

        //   cout << "intensity at specular= " << intensity <<  "\n";
        // }


        test.RecordBrdf(base::Vec3f(wi[0], wi[1], wi[2]),
                        base::Vec3f(w0[0], w0[1], w0[2]),
                        base::Vec3f(intensity[2], intensity[1], intensity[0]),
                        normal, tangent, bitangent);
      }
    }

    imwrite("/home/stratman/Documents/code_public/light_stage_test/"
            "brdf_debug_pics/angles_" +
                std::to_string(elem) + ".exr",
            brdfdebug_angles);

    imwrite("/home/stratman/Documents/code_public/light_stage_test/"
            "brdf_debug_pics/wi/wi_" +
                std::to_string(elem) + ".exr",
            wi_angles);

    imwrite("/home/stratman/Documents/code_public/light_stage_test/"
            "brdf_debug_pics/wo/wo_" +
                std::to_string(elem) + ".exr",
            w0_angles);

   imwrite("/home/stratman/Documents/code_public/light_stage_test/"
            "brdf_debug_pics/normals/n_" +
                std::to_string(elem) + ".exr",
            normals);

    // colorize found circle and line to highest peak in output image
    line(img, Point(max_loc.x, max_loc.y),
         Point(center.x - offset.x, center.y - offset.y), Scalar(255, 0, 0), 1,
         8, 0);
    circle(img, Point(center.x - offset.x, center.y - offset.y), radius,
           Scalar(0, 0, 255), 3, 10, 0);
    imwrite("/home/stratman/Documents/code_public/light_stage_test/"
            "detectcirclestest/detected_" +
                std::to_string(elem) + ".exr",
            img);
    
    // circle(img, center, radius, Scalar(0, 0, 255), 2, 8, 0);
    locations.write((char *)&lightlocation, sizeof(cv::Vec3f));
    Vec3f ll_dir = (-(reflection_vector));
    locations.write((char *)&ll_dir, sizeof(cv::Vec3f));

    // bfile.close();
    // wo.close();
    // wI.close();
    

    cout << "Picture " << std::to_string(elem)
         << " has been processed and filled up unto " << std::to_string(idx)
         << "\n";
  }
  wi_angles_bins.close();
  locations.close();
  test.filter(0.25f);

     static const unsigned char mix_with[] = {1, 20, 128,191};

  //     for (unsigned char elem2 : mix_with) {
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(stop - start);

  cout << "Duration of the Acquisition Process is: " << duration.count() << endl;

  cout << "========= Re-evaluation process started ============== \n";
  // bool mixwi = false;
  String toevaluate = "20";
  //String tomixwith = std::to_string(elem2);

 // for (int idx = 0; idx < 1400; idx = idx + 10) {
 for (unsigned char elem1 : upper_hemi) {
  //reevaluate
  Mat brdf_reevaluate(Size(1400,1400), CV_32FC3, 0.0);

  Mat wo =
  imread("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/wo/wo_"+  to_string(elem1) + ".exr",
  IMREAD_UNCHANGED );

  Mat wi =
  imread("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/wi/wi_"+  to_string(elem1) + ".exr",
  IMREAD_UNCHANGED );

  // Mat wi_alternate =
  // imread("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/wi/wi_"+ tomixwith + ".exr", IMREAD_UNCHANGED );

  Mat normal =
  imread("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/normals/n_"+  to_string(elem1) + ".exr",
  IMREAD_UNCHANGED );

  // for arbitrary light position
  /*
      Vec3f max_loc_3d_dir =
        generateRay(Point(idx + offset.x, center.y), 1300.0,
                    500.0, 1400.0, 1400.0, 3000.0, 4096.0, fovx, fovy);

    Vec3f max_loc_3d =
        CameraOrigin +
        max_loc_3d_dir * ray_sphere_intersec(max_loc_3d_dir, CameraOrigin,
                                             CenterPoint, sphere_radius);

    Vec3f max_loc_normal = normalize(max_loc_3d - CenterPoint);

    Vec3f reflection_vector =
        normalize((max_loc_3d_dir - (2 * (max_loc_3d_dir.dot(max_loc_normal)) *
                                     max_loc_normal)));

    Vec3f lightlocation = max_loc_3d + reflection_vector * distance_cam_sphere_alternative;
    */

   for (int i = 0; i < wo.rows; ++i) {
      for (int j = 0; j < wo.cols; ++j) {

    Vec3f wi_vec= wi.at<Vec3f>(i, j);

  // Vec3f wi_alt_vec= wi_alternate.at<Vec3f>(i, j);
      Point arb(j + offset.x, i + offset.y);
      Vec3f arbi_3DPoint_dir = generateRay(arb, 1300.0, 500.0, 1400.0, 1400.0,
                                             3000.0, 4096.0, fovx, fovy);

        // then, with the distance calculated from the ray-sphere trace we get
        // the vector arbi_3DPoint
        Vec3f arbi_3DPoint =
            CameraOrigin +
            arbi_3DPoint_dir * ray_sphere_intersec(arbi_3DPoint_dir,
                                                   CameraOrigin, CenterPoint,
                                                   sphere_radius);

        // if the 3dPoint is at 0,0,0 the distance of the ray sphere intersect
        // is 0, which means the ray-sphere-intersect returns 0 as in it doesnt
        // hit the sphere should result in the similar
        if (arbi_3DPoint == Vec3f(0.f, 0.f, 0.f)) {
          // cout << "arbi3dPoint is 0 ,0 ,0 !!! \n";
          continue;
        }



   // Vec3f wi_vec = normalize(lightlocation - arbi_3DPoint);

  Vec3f wo_vec= wo.at<Vec3f>(i, j);
  Vec3f normal_vec= normal.at<Vec3f>(i, j);

  if((wi_vec[0] == 0.f && wi_vec[1] == 0.f && wi_vec[2] == 0.f) || (wo_vec[0] == 0.f && wo_vec[1] == 0.f && wo_vec[2] == 0.f)){
    continue;
  }
  base::Vec3f wi_vec_base(wi_vec[2],wi_vec[1],wi_vec[0]);
  // base::Vec3f wi_vec_alt_base(wi_alt_vec[2],wi_alt_vec[1],wi_alt_vec[0]);
  base::Vec3f wo_vec_base(wo_vec[2],wo_vec[1],wo_vec[0]);
  base::Vec3f normal_base(normal_vec[2],normal_vec[1],normal_vec[0]);

//   if(mixwi){
//   wi_vec_base = (wi_vec_alt_base + wi_vec_base).normalize();
// }
  base::Vec3f px_value = test.DebugBrdf(wi_vec_base,wo_vec_base,normal_base);

  brdf_reevaluate.at<Vec3f>(i, j) = Vec3f(px_value[0], px_value[1], px_value[2]);
      }
   }

  // if(mixwi){
  // imwrite("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/reevaluated/reeval_" + toevaluate + "-" + tomixwith + ".exr",
  //           brdf_reevaluate);
  //           }else {
  imwrite("/home/stratman/Documents/code_public/light_stage_test/brdf_debug_pics/reevaluated/reeval_"+  to_string(elem1) + ".exr",
            brdf_reevaluate);
            // }
  //     }
   }


  test.save(
      "/home/stratman/Documents/code_public/light_stage_test/brdf.binary");
  return EXIT_SUCCESS;
}
