#include <CameraDefs.h>
#include <ImfCompression.h>
#include <ImfPixelType.h>
#include <LightStage/LightStage.hh>
#include <SpinnakerCamera/CaptureConfig.h>
#include <SpinnakerCamera/ImageProcessor.h>
#include <SpinnakerCamera/ImageReader.h>
#include <SpinnakerCamera/OptHDR.h>
#include <SpinnakerCamera/Pipeline.h>
#include <SpinnakerCamera/PipelineWorker.h>
#include <SpinnakerDefs.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/utility.hpp>
#include <string>
#include <unistd.h>
#include <chrono>


#include <SpinnakerCamera/sinks/ExrWriter.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

//#include <detectCircles.h>

using namespace SpinCam;
using namespace light_stage;
using namespace std;
using namespace cv;
using namespace std::chrono;

int main() {

  LightStage ls;
  ls.init();
  ls.clearAll();
  ls.upload();
  // static const unsigned char upper_hemi[] = {
  //     7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
  //     21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,
  //     35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
  //     60,  61,  62,  63,  64,  65,  66,  67,  68,  74,  75,  76,  77,  78,
  //     79,  80,  89,  92,  93,  94,  95,  96,  97,  101, 102, 103, 104, 105,
  //     116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130,
  //     131, 132, 133, 134, 147, 148, 149, 150, 151, 152, 153, 157, 158, 159,
  //     160, 161, 162, 163, 164, 165, 166, 171, 172, 173, 174, 175, 176, 177,
  //     178, 179, 180, 181, 187, 188, 190, 192, 195};

  static const unsigned char new_hemi[] = {
      1
      ,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
      16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,
      32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
      47,  48,  49,  50,  51,  52,  53,  54,
      
       59,  60,  61,  62,  63,  64,  65,
      66,  67,  68,  69,  73,  74,  75,  76,  77,  78,  79,  80,  89,  92,  93,
      94,  95,  96,  97,  101, 102, 103, 104, 105, 106, 110, 115, 116, 117, 118,
      119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
      134, 135, 145, 146, 147, 148, 149, 150, 151, 152, 153, 156, 157, 158, 159,
      160, 161, 162, 163, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 176,
      177, 178, 179, 180, 181, 185, 187, 188, 189, 190, 191, 192, 195
      };

  // static const unsigned char main_arc[] = {
  //     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
  //     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  //     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
  //     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};

  // INIT OPT HDR

  OptHDR hdrnode("/home/stratman/Documents/code_public/light_stage_test/"
                 "Opt_HDR_calibration.json",
                 RoI(1350, 300, 1600, 1600),
                 "/home/stratman/Documents/code_public/light_stage_test/"
                 "pipeline_mean.json");

  CaptureConfig hdr_cc;
  std::vector<CaptureConfig> exposure_series = {
      CaptureConfig(5000, 2),   CaptureConfig(15000, 2),
      CaptureConfig(30000, 2),  CaptureConfig(60000, 2),
      CaptureConfig(120000, 2), CaptureConfig(240000, 2),
  };

  auto exposure_series_ptr =
      std::make_shared<std::vector<CaptureConfig>>(exposure_series);
  hdr_cc.SetExposureSeries(exposure_series_ptr);

  hdrnode.setCaptureConfig(hdr_cc);
  // for every led light, a darkframe and ldr image (exposure series) is taken
  // the df is taken first so the lights can be toggled on after that

  // Capture the necessary Darkframes
  hdrnode.captureDarkFrames();

  std::cout << "Chessboard-Configuration started \n";

  //===================Camera Configuration
  //(Chessboard)===================================

  // Defining the dimensions of checkerboard
  int CHECKERBOARD[2]{11, 23};

  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f>> objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f>> imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for (int i{0}; i < CHECKERBOARD[1]; i++) {
    for (int j{0}; j < CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j, i, 0));
  }

  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string path = "/home/stratman/Documents/code_public/light_stage_test/"
                     "checkerboard_configframes/*.png";

  cv::glob(path, images);

  std::cout << "Checkerboard Images successfully loaded \n";

  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the directory
  for (int i{0}; i < images.size(); i++) {
    frame = cv::imread(images[i]);

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true
    success = cv::findChessboardCorners(
        gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK |
            CALIB_CB_NORMALIZE_IMAGE);

    std::cout << fmt::format("findChessboardCorners was successful? {} \n",
                             success);

    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
     */
    if (success) {
      cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30,
                                0.001);

      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1),
                       criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame,
                                cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
                                corner_pts, success);

      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }

    // cv::imshow("Image",frame);
    // cv::waitKey(0);
  }

  // cv::destroyAllWindows();

  cv::Mat cameraMatrix, distCoeffs, R, T;

  /*
   * Performing camera calibration by
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the
   * detected corners (imgpoints)
   */

  cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols),
                      cameraMatrix, distCoeffs, R, T);

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
  //=================================== END CHESSBOARD CONFIG
  //======================================= iteraters through every light board
  // and undistorts, crops, demosaiced it
  auto start = high_resolution_clock::now();

  for (size_t elem : new_hemi) {

    // Capture the ldr Images
    ls.setLedOnBoard(elem, 15, 255);
    ls.upload();
    hdrnode.captureExposureSerie();
    hdrnode.setMaxIterations(30);

    Image HDR_img = hdrnode.computeHDR();

    ExrWriter writer(
        "/home/stratman/Documents/code_public/light_stage_test/HDR_pics",
        Imf::PIZ_COMPRESSION, Imf::FLOAT);

    CaptureConfig hdr_config = hdrnode.getCaptureConfig();

    hdr_config.SetFileName("hdr_image_" + std::to_string(elem));
    writer.saveImage(HDR_img, hdr_config);

    // DEMOSAICING
    cv::Mat img = imread("/home/stratman/Documents/code_public/"
                         "light_stage_test/HDR_pics/hdr_image_" +
                             std::to_string(elem) + ".exr",
                         cv::IMREAD_UNCHANGED);

    // the focal length is set to 10248 so it is the same on both axis
    Mat newcameramtx = (Mat_<double>(3, 3) << 10248, 0, 2021.360907134241, 0,
                        10248, 1477.123308582499, 0, 0, 1);
    // cameraMatrix[0][2] = cameraMatrix[0][2] - (500 + 700);
    // cameraMatrix[1][2] = cameraMatrix[1][2] - (1550 + 700);

    Mat newcammtx_offset =
        (Mat_<double>(3, 3) << 10247.46054514974, 0,
         2021.360907134241 - (500 + 700), 0, 10248.45719903782,
         1477.123308582499 - (1300 + 700), 0, 0, 1);

    Mat mapx, mapy;

    cout << cameraMatrix << "\n";
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
                                newcameramtx, cv::Size(1400, 1400), CV_16SC2,
                                mapx, mapy);

    cout << "Rectify Map was created successfully"
         << "\n";

    cv::Mat undist;
    cv::remap(img, undist, mapx, mapy, INTER_LINEAR);
    mapx.release();
    mapy.release();
    img.release();

    cout << "Image was successfully remapped"
         << "\n";

    // float scale = 1000; //(1 << (16 - 12)) - 1.f; // from 12 bit to 16 bit

    // cv::Mat frac = cv::Mat(undist.rows, undist.cols, CV_32F);

    // // Get fractional part of the pixel values
    // for (int i = 0; i < undist.rows; ++i) {
    //   for (int j = 0; j < undist.cols; ++j) {

    //     float intensity = undist.at<float>(i, j);

    //     float fraction = intensity - floor(intensity);

    //     frac.at<float>(i, j) = fraction * scale;
    //     undist.at<float>(i, j) = floor(intensity);
    //   }
    // }

    // convert undistorted image to integers for demosaicing
    // std::cout << "undist(0,0) before converting to int: "
    //           << undist.at<float>(0, 0) << " , has the type: " << undist.type()
    //           << "\n";
    cv::Mat undistUint16 = cv::Mat(undist.rows, undist.cols, CV_16U);
    undist.convertTo(undistUint16,
                     CV_16U); // std::numeric_limits<uint16_t>::max());
    undist.release();
    // std::cout << "undist(0,0) after converting to int: "
    //           << undistUint16.at<unsigned int>(0, 0)
    //           << " , has the type: " << undistUint16.type() << "\n";

    // // convert fractional part of undist (scale beforehand) to integers for
    // // demosaicing
    // std::cout << "Frac(0,0) before converting to int: " << frac.at<float>(0, 0)
    //           << " , has the type: " << frac.type() << "\n";
    // cv::Mat fracUint16 = cv::Mat(frac.rows, frac.cols, CV_16U);

    // frac.convertTo(fracUint16, CV_16U);

    // frac.release();

    // std::cout << "Frac(0,0) after converting to int: "
    //           << fracUint16.at<unsigned int>(0, 0)
    //           << " , has the type: " << fracUint16.type() << "\n";

    // cout << "undistorted image and its fractional part successfully converted"
    //      << "\n";

    cv::Mat demosaiced =
        cv::Mat(undistUint16.rows, undistUint16.cols, CV_16UC3);
    // cv::Mat frac_demosaiced =
    //     cv::Mat(fracUint16.rows, fracUint16.cols, CV_16UC3);
    cv::demosaicing(undistUint16, demosaiced, cv::COLOR_BayerRG2BGR, 3);
    // cv::demosaicing(fracUint16, frac_demosaiced, cv::COLOR_BayerRG2BGR, 3);
    // COLOR_BayerGR2BGR, COLOR_BayerRG2BGR, COLOR_BayerGB2BGR

    cout << "Image and its fractional part was successfully demosaiced "
         << "\n";

    cv::Mat out = cv::Mat(demosaiced.rows, demosaiced.cols, CV_32FC3);
    // cv::Mat frac_out =
    //     cv::Mat(frac_demosaiced.rows, frac_demosaiced.cols, CV_32FC3);

    demosaiced.convertTo(out, CV_32FC3);
    // frac_demosaiced.convertTo(frac_out, CV_32FC3);

    // for (int i = 0; i < frac_out.rows; ++i) {
    //   for (int j = 0; j < frac_out.cols; ++j) {
    //     Vec3f intensity1 = frac_out.at<Vec3f>(i, j);
    //     Vec3f intensity2 = out.at<Vec3f>(i, j);
    //     Vec3f intensity3 = (intensity1 / scale) + intensity2;

    //     out.at<Vec3f>(i, j) =
    //         Vec3f(intensity3[2], intensity3[1], intensity3[0]);
    //   }
    // }

    // temp = temp(Range(500, 1900) ,Range(1550,2950));

    imwrite("/home/stratman/Documents/code_public/light_stage_test/HDR_pics/"
            "hdr_image_final_" +
                std::to_string(elem) + ".exr",
            out);
    ls.clearAll();
    ls.upload();
  }


  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Duration of the Acquisition Process is: " << duration.count() << endl;


// To get the value of duration use the count()
// member function on the duration object
  // sleep(60);
  // future.get();
  ls.clearAll();
  ls.upload();

  // ls.arc->moveTo(0);
}
