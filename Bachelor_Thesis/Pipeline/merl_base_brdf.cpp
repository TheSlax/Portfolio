#include "merl_base_brdf.h"
#include "MatrixNxN.hh"
#include "Vector.hh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <opencv2/core.hpp>
#include <spdlog/pattern_formatter-inl.h>
#include <stdio.h>
//#include <vcruntime.h>
#include <fstream>
#include <string>
#include <vector>

MerlBaseBrdf::MerlBaseBrdf(const std::string &path) {

  // reads file, checks if valid
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "Could not open file '%s'.\n", path.c_str());
    valid_ = false;
    return;
  }

  // checks dimensions of the brdf data
  unsigned int dims[3];
  size_t read = fread(dims, sizeof(int), 3, f);
  if (read != 3) {
    fprintf(stderr, "Could not read dimensions of brdf data.\n");
    fclose(f);
    valid_ = false;
    return;
  }

  // checks if the size of the resolution of the bins is the same
  unsigned int n = dims[0] * dims[1] * dims[2];
  if (n != k_sampling_res_theta_h * k_sampling_res_theta_d *
               k_sampling_res_phi_d / 2) {
    fprintf(stderr, "Dimensions don't match\n");
    fclose(f);
    valid_ = false;
    return;
  }

  brdf_ = new double[n * 3];
  // fills up all spaces with 0
  for (unsigned int i = 0; i < n * 3; i++) {
    brdf_[i] = 0;
  }

  brdf_proposal.resize(n * 3);
  read = fread(brdf_, sizeof(double), 3 * n, f);
  fclose(f);
  if (read == 3 * n) {
    valid_ = true;
  } else {
    delete[] brdf_;
    valid_ = false;
  }
}

MerlBaseBrdf::MerlBaseBrdf() {
  size_t n = k_sampling_res_theta_h * k_sampling_res_theta_d *
             k_sampling_res_phi_d / 2;
  brdf_ = new double[n * 3];
  brdf_proposal.resize(n * 3);
}

MerlBaseBrdf::~MerlBaseBrdf() {
  if (valid_)
    delete[] brdf_;
}

base::Vec2f
MerlBaseBrdf::cartesian2sphereCoordNormalized(const base::Vec3f &d) const {
  base::Vec3f a = d;
  // return base::Vec2f(atan2(hypot(d[0] , d[2]) , d[1]), atan2(d[0], d[2]));
  // base::Vec2f(acos(d[2]), atan2(d[1], d[0]));
  ////base::Vec2f(atan2(d[1],d[0]) , atan2(sqrt(d[0]*d[0]+d[1]*d[1]),d[2]));
  /////theta , phi

  return base::Vec2f(acos(a[2] / a.norm()), atan2(a[1], a[0]));
}

base::Vec3f MerlBaseBrdf::rotateVector(const base::Vec3f &v,
                                       const base::Vec3f &axis,
                                       float angle) const {
  float cos_ang = cos(angle);
  base::Vec3f result = v;
  result *= cos_ang;
  float tmp = (v.dot(axis)) * (1.f - cos_ang);
  result += axis * tmp;
  result += axis.cross(v) * sin(angle);
  return result;

  // code from
  // https://stackoverflow.com/questions/42421611/3d-vector-rotation-in-c
  // float cos_ang = cos(angle);
  // float sin_ang = sin(angle);
  // base::Vec3f result = (v * cos_ang) + (cross(axis, v) * sin_ang) + (axis *
  // dot(axis, v)) * (1 - cos_ang); return result;
}

base::Vec4f MerlBaseBrdf::stdCoordsToHalfDiffCoords(
    const base::Vec3f &w_i, const base::Vec3f &w_o, const base::Vec3f &norm,
    const base::Vec3f &bi_norm) const {
  // compute halfway vector
  base::Vec3f half = (w_i + w_o) / 2.f;
  half.normalize();

  // compute  (theta_half, fi_half)
  base::Vec2f tp_half = cartesian2sphereCoordNormalized(half);

  // base::Vec3f bi_normal(0, 1, 0);
  // base::Vec3f normal(0, 0, 1);
  base::Vec3f temp = rotateVector(w_i, norm, -tp_half[1]);
  base::Vec3f diff = rotateVector(temp, bi_norm, -tp_half[0]);

  base::Vec2f tp_diff = cartesian2sphereCoordNormalized(diff);
  return base::Vec4f(tp_half[0], tp_half[1], tp_diff[0], tp_diff[1]);
}

void MerlBaseBrdf::save(const std::string &path) {

  std::fstream bfile;
  int dims[3];
  dims[0] = k_sampling_res_theta_h;
  dims[1] = k_sampling_res_theta_d;
  dims[2] = k_sampling_res_phi_d / 2;

  int n = k_sampling_res_theta_h * k_sampling_res_theta_d *
          k_sampling_res_phi_d / 2;

  bfile = std::fstream(path, std::ios::out | std::ios::binary);

  bfile.write(reinterpret_cast<char *>(dims), sizeof(int) * 3);
  bfile.write(reinterpret_cast<char *>(brdf_), sizeof(double) * n * 3);

  bfile.close();
}

base::Vec3f MerlBaseBrdf::Evaluate(const base::Vec3f &w_i,
                                   const base::Vec3f &w_o) {
  if (!valid_)
    return base::Vec3f(0, 0, 0);

  base::Vec4d half_diff = stdCoordsToHalfDiffCoords(
      w_i, w_o, base::Vec3f(0, 0, 1), base::Vec3f(0, 1, 0));

  int ind = (k_sampling_res_theta_h * k_sampling_res_theta_d *
             k_sampling_res_phi_d / 2) -
            (phiDiffIndex(half_diff[3]) +
             thetaDiffIndex(half_diff[2]) * k_sampling_res_phi_d / 2 +
             thetaHalfIndex(half_diff[0]) * k_sampling_res_phi_d / 2 *
                 k_sampling_res_theta_d);

  // Find index.
  // Note that phi_half is ignored, since isotropic BRDFs are assumed
  size_t red = ind;
  size_t green = ind + k_sampling_res_theta_h * k_sampling_res_theta_d *
                           k_sampling_res_phi_d / 2;
  size_t blue = ind + k_sampling_res_theta_h * k_sampling_res_theta_d *
                          k_sampling_res_phi_d;

  return base::Vec3f(brdf_[red] * k_red_scale, brdf_[green] * k_green_scale,
                     brdf_[blue] * k_blue_scale);
}
base::Vec3f MerlBaseBrdf::DebugBrdf(const base::Vec3f &w_i,
                                    const base::Vec3f &w_o,
                                    const base::Vec3f &normal) {

  base::Vec3f forward(0, 0, 1);
  base::Vec3f up(0, -1, 0);
  base::Vec3f right(1, 0, 0);

float m[9];
  m[0] = right[0];
  m[3] = right[1];
  m[6] = right[2];

  m[1] = up[0];
  m[4] = up[1];
  m[7] = up[2];

  m[2] = forward[0];
  m[5] = forward[1];
  m[8] = forward[2];

  base::MatrixNxN<float, 3> m_change_to(m);
  base::Vec3f w_i_global = m_change_to * w_i;
  base::Vec3f w_o_global = m_change_to * w_o;

  //base::Vec3f norm(0,1,0);
  base::Vec3f tangent(1,0,0);
  base::Vec3f binorm = cross(tangent, normal);
  tangent = cross(binorm, normal);

  base::Vec3f H = (w_i + w_o).normalize();
  

  float theta_H = acos(
      std::clamp(dot(normal, H), 0.f, 1.f)); //= acos(clamp(dot(norm, H), 0, 1));
  float theta_diff = acos(std::clamp(dot(H, w_i), 0.f, 1.f));
  // std::cout << "theta_diff: " << std::to_string(theta_diff) << " =
  // acos(clamp(" << std::to_string(dot(H, w_i)) << ", 0, 1))" << "\n";
  float phi_diff = 0;

  if (theta_diff < 1e-3f) {
    // phi_diff indeterminate, use phi_half instead
    phi_diff = atan2(std::clamp(-dot(w_i, binorm), -1.f, 1.f),
                     std::clamp(dot(w_i, tangent), -1.f, 1.f));

    // atan(clamp(-dot(toLight, bitangent), -1, 1), clamp(dot(toLight, tangent),
    // -1, 1));

  } else if (theta_H > 1e-3f) {
    // use Gram-Schmidt orthonormalization to find diff basis vectors
    base::Vec3f u = -(normal - dot(normal, H) * H).normalize();
    base::Vec3f v = cross(H, u);
    phi_diff = atan2(std::clamp(dot(w_i, v), -1.f, 1.f),
                     std::clamp(dot(w_i, u), -1.f, 1.f));
  } else {
    theta_H = 0;
  }

  int ind = phiDiffIndex(phi_diff) +
            thetaDiffIndex(theta_diff) * k_sampling_res_phi_d / 2 +
            thetaHalfIndex(theta_H) * k_sampling_res_phi_d / 2 *
                k_sampling_res_theta_d;

  // Find index.
  // Note that phi_half is ignored, since isotropic BRDFs are assumed
  size_t red = ind;
  size_t green = ind + k_sampling_res_theta_h * k_sampling_res_theta_d *
                           k_sampling_res_phi_d / 2;
  size_t blue = ind + k_sampling_res_theta_h * k_sampling_res_theta_d *
                          k_sampling_res_phi_d;



  return base::Vec3f(brdf_[red], brdf_[green], brdf_[blue]);
}
void MerlBaseBrdf::RecordBrdf(const base::Vec3f &w_i, const base::Vec3f &w_o,
                              const base::Vec3f &px_value,
                              const base::Vec3f &norm,
                              const base::Vec3f &tangent,
                              const base::Vec3f &bitangent) {

  base::Vec3f forward(0, 0, 1);
  base::Vec3f up(0, -1, 0);
  base::Vec3f right(1, 0, 0);

  float m[9];
  m[0] = right[0];
  m[3] = right[1];
  m[6] = right[2];

  m[1] = up[0];
  m[4] = up[1];
  m[7] = up[2];

  m[2] = forward[0];
  m[5] = forward[1];
  m[8] = forward[2];

  base::MatrixNxN<float, 3> m_change_to(m);
  base::Vec3f w_i_global = m_change_to * w_i;
  base::Vec3f w_o_global = m_change_to * w_o;
  base::Vec3f tangent_global = m_change_to * tangent;
  base::Vec3f normal_global = m_change_to * norm;
  base::Vec3f bitangent_global = m_change_to * bitangent;

  float m_local[9];
  m_local[0] = tangent_global[0];
  m_local[3] = tangent_global[1];
  m_local[6] = tangent_global[2];

  m_local[1] = normal_global[0];
  m_local[4] = normal_global[1];
  m_local[7] = normal_global[2];

  m_local[2] = bitangent_global[0];
  m_local[5] = bitangent_global[1];
  m_local[8] = bitangent_global[2];

  base::MatrixNxN<float, 3> m_change_to_local(m_local);
  base::Vec3f w_i_local = m_change_to_local * w_i;
  base::Vec3f w_o_local = m_change_to_local * w_o;

  /**/
  // fill base with vectors
  //  float matrixData[9];
  //   matrixData[0] = bitangent[0];
  //   matrixData[3] = bitangent[1];
  //   matrixData[6] = bitangent[2];

  //   matrixData[1] = tangent[0];
  //   matrixData[4] = tangent[1];
  //   matrixData[7] = tangent[2];

  //   matrixData[2] = norm[0];
  //   matrixData[5] = norm[1];
  //   matrixData[8] = norm[2];

  /*
    //base constructed now calculate determinant and invert
    // https://mo.mathematik.uni-stuttgart.de/inhalt/beispiel/beispiel1113/
    float determinant = matrixData[0] * matrixData[4] * matrixData[8] +
                        matrixData[1] * matrixData[5] * matrixData[6] +
                        matrixData[2] * matrixData[3] * matrixData[7] -
                        matrixData[1] * matrixData[3] * matrixData[8] -
                        matrixData[2] * matrixData[4] * matrixData[6] -
                        matrixData[0] * matrixData[5] * matrixData[7];

    float invdet = 1 / determinant;

    matrixData[0] = (matrixData[4] * matrixData[8] - matrixData[5] *
   matrixData[7]) * invdet; matrixData[1] = (matrixData[2] * matrixData[7] -
   matrixData[1] * matrixData[8]) * invdet; matrixData[2] = (matrixData[1] *
   matrixData[5] - matrixData[2] * matrixData[4]) * invdet; matrixData[3] =
   (matrixData[5] * matrixData[6] - matrixData[3] * matrixData[8]) * invdet;
    matrixData[4] = (matrixData[0] * matrixData[8] - matrixData[2] *
   matrixData[6]) * invdet; matrixData[5] = (matrixData[3] * matrixData[2] -
   matrixData[0] * matrixData[5]) * invdet; matrixData[6] = (matrixData[3] *
   matrixData[7] - matrixData[6] * matrixData[4]) * invdet; matrixData[7] =
   (matrixData[6] * matrixData[1] - matrixData[0] * matrixData[7]) * invdet;
    matrixData[8] = (matrixData[0] * matrixData[4] - matrixData[3] *
   matrixData[1]) * invdet;

   float matrixDatacolumfirst[9];

   matrixDatacolumfirst[0] = matrixData[0];
   matrixDatacolumfirst[1] = matrixData[3];
   matrixDatacolumfirst[2] = matrixData[6];
   matrixDatacolumfirst[3] = matrixData[1];
   matrixDatacolumfirst[4] = matrixData[4];
   matrixDatacolumfirst[5] = matrixData[7];
   matrixDatacolumfirst[6] = matrixData[2];
   matrixDatacolumfirst[7] = matrixData[5];
   matrixDatacolumfirst[8] = matrixData[8];
   */
  //  matrixDatacolumfirst[0] = bitangent[0];
  //  matrixDatacolumfirst[1] = bitangent[1];
  //  matrixDatacolumfirst[2] = bitangent[2];
  //  matrixDatacolumfirst[3] = tangent[0];
  //  matrixDatacolumfirst[4] = tangent[1];
  //  matrixDatacolumfirst[5] = tangent[2];
  //  matrixDatacolumfirst[6] = norm[0];
  //  matrixDatacolumfirst[7] = norm[1];
  //  matrixDatacolumfirst[8] = norm[2];

  // TODO change of variables paper
  // https://www.cs.princeton.edu/~smr/papers/brdf_change_of_variables/brdf_change_of_variables.pdf
  /**/
  // base::Vec3f w_i_local = base::Vec3f(dot(w_i, tangent), dot(w_i, norm),
  // dot(w_i, bitangent));
  // float wi_x = dot(w_i, tangent);
  // float wi_y = dot(w_i, norm);
  // float wi_z = dot(w_i, bitangent);
  // base::Vec3f  w_i_local = wi_x * tangent + wi_y * norm + wi_z * bitangent;

  // //base::Vec3f w_o_local = base::Vec3f(dot(w_o, tangent), dot(w_o, norm),
  // dot(w_o, bitangent)); float wo_x = dot(w_o, tangent); float wo_y = dot(w_o,
  // norm); float wo_z = dot(w_o, bitangent); base::Vec3f  w_o_local = wo_x *
  // tangent + wo_y * norm + wo_z * bitangent;

  //   base::Vec3f H_local = (w_i_local + w_o_local).normalize(); //Equation (7)

  //   base::Vec2f H_local_spherical = cartesian2sphereCoordNormalized(H_local);
  //   //spherical Coordinates of h

  //   base::Vec3f diff = rotateVector(rotateVector(w_i_local,
  //   base::Vec3f(0,1,0), -H_local_spherical[1]),
  //                                                           base::Vec3f(1,0,1),
  //                                                           -H_local_spherical[0]);
  //                                                           //Equation (9)
  //                                                           bit
  //                                                           base::Vec3f(0.f,
  //                                                           0.f, 1.f) norm
  //                                                           base::Vec3f(0.f, 1.f,
  //                                                           0.f)
  //   base::Vec2f diff_spherical = cartesian2sphereCoordNormalized(diff);
  //   //spherical Coordinates of d

  //   // =================== Indices =============

  // size_t ind =phiDiffIndex(diff_spherical[1]) +
  //             thetaDiffIndex(diff_spherical[0]) * k_sampling_res_phi_d / 2 +
  //             thetaHalfIndex(H_local_spherical[0]) * k_sampling_res_phi_d / 2
  //             * k_sampling_res_theta_d;

  // std::cout << "phi diff: " << std::to_string(phiDiffIndex(diff_spherical[1])
  // )
  //         << " theta_diff: " <<
  //         std::to_string(thetaDiffIndex(diff_spherical[0]))
  //            <<" theta_H: " <<
  //            std::to_string(thetaHalfIndex(H_local_spherical[0]))   <<"\n";

  // =================== TESTS =============
  // if ((abs(w_i.norm() - 1.f) > 1e-3) || (abs(w_i_local.norm() - 1.f) > 1e-3))
  // { std::cout << "(" << w_i.norm() << ", " << w_i_local.norm() << ")";
  // }
  // if(isnan(diff_spherical[0])){
  //   cout << " assert fails here" << "\n";
  // }
  //  if(H_local_spherical[0] > (M_PI / 2.0)){
  //   cout << " assert fails here" << "\n";
  // }
  // assert(!(isnan(diff_spherical[0])));
  // assert(!(isnan(diff_spherical[1])));
  // assert(!(isnan(H_local_spherical[0])));
  // assert(!(isnan(H_local_spherical[1])));
  // assert(!((abs(w_i_local.norm() - 1.f) > 1e-3)));
  // //assert(!((abs(diff.norm() - 1.f) > 1e-3)));
  // assert(ind < k_sampling_res_theta_h * k_sampling_res_theta_d *
  // k_sampling_res_phi_d);
  // assert(rotateVector(base::Vec3f(1.f,0.f,0.f), base::Vec3f(0.f,0.f,1.f),
  // M_PI) == base::Vec3f(-1.f,0.f,0.f));
  // assert(cartesian2sphereCoordNormalized(base::Vec3f(1.f,0.f,0.f)) ==
  // base::Vec2f(0.f,static_cast<float>(M_PI / 2.0)));
  // assert(cartesian2sphereCoordNormalized(base::Vec3f(0.f,1.f,0.f)) ==
  // base::Vec2f(static_cast<float>(M_PI / 2.0),static_cast<float>(M_PI / 2.0)));
  // assert(cartesian2sphereCoordNormalized(base::Vec3f(0.f,1.f,1.f)) ==
  // base::Vec2f(static_cast<float>(M_PI / 2.0),static_cast<float>(M_PI / 4.0)));

  /*if(cartesian2sphereCoordNormalized(base::Vec3f(0.f,1.f,1.f)) ==
  base::Vec2f(static_cast<float>(M_PI / 2.0),static_cast<float>(M_PI / 4.0))){
    std::cout << "cartesian2sphereCoordNormalized() does not work as intended,
  is: "
    << cartesian2sphereCoordNormalized(base::Vec3f(0.f,1.f,1.f))
    << " \n";
  }

  if (rotateVector(base::Vec3f(1.f,0.f,0.f), base::Vec3f(0.f,0.f,1.f), M_PI) ==
  base::Vec3f(-1.f,0.f,0.f)){ std::cout << "rotateVector() does not work as
  intended, is: "
    << rotateVector(base::Vec3f(1.f,0.f,0.f), base::Vec3f(0.f,0.f,1.f), M_PI)
    << " \n";
  } */

  // base::Vec3f w_i_local = base::Vec3f(dot(w_i, tangent), dot(w_i, norm),
  // dot(w_i, bitangent)); base::Vec3f w_o_local = base::Vec3f(dot(w_o,
  // tangent), dot(w_o, norm), dot(w_o, bitangent));
  base::Vec3f H = (w_i + w_o).normalize();
  // base::Vec3f tangent_ortho = cross(bitangent, norm);

  // base::Vec2f H_local_spherical = cartesian2sphereCoordNormalized(H);
  float theta_H =
      acos(std::clamp(dot(norm, H), 0.f, 1.f)); // H_local_spherical[0];
  // base::Vec3f diff = rotateVector(rotateVector(w_i_local, normal,
  // -H_local_spherical[1]), bitangent, -H_local_spherical[0]); base::Vec2f
  // diff_spherical = cartesian2sphereCoordNormalized(diff);

  // diff[0];
  //   base::Vec3f diff = (w_i_local - w_o_local).normalize();
  //   base::Vec2f diff_spherical =  cartesian2sphereCoordNormalized(diff);
  // diff_spherical[0];
  float theta_diff = acos(std::clamp(dot(H, w_i), 0.f, 1.f));
  // std::cout << "theta_diff: " << std::to_string(theta_diff) << "\n" ;
  // acos(clamp(" << std::to_string(dot(H, w_i)) << ", 0, 1))" << "\n";
  float phi_diff = 0;

  if (theta_diff < 1e-3f) {
    // phi_diff indeterminate, use phi_half instead
    phi_diff = atan2(std::clamp(-dot(w_i, bitangent), -1.f, 1.f),
                     std::clamp(dot(w_i, tangent), -1.f, 1.f));

    // atan(clamp(-dot(toLight, bitangent), -1, 1), clamp(dot(toLight, tangent),
    // -1, 1));

  } else if (theta_H > 1e-3f) {
    // use Gram-Schmidt orthonormalization to find diff basis vectors
    base::Vec3f u = -(norm - dot(norm, H) * H).normalize();
    base::Vec3f v = cross(H, u);
    phi_diff = atan2(std::clamp(dot(w_i, v), -1.f, 1.f),
                     std::clamp(dot(w_i, u), -1.f, 1.f));
  } else {
    theta_H = 0;
  }

  // phi_diff = diff_spherical[1];

  int ind = phiDiffIndex(phi_diff) +
            thetaDiffIndex(theta_diff) * k_sampling_res_phi_d / 2 +
            thetaHalfIndex(theta_H) * k_sampling_res_phi_d / 2 *
                k_sampling_res_theta_d;

  indices_arr[ind]++;

  phi_diff_arr[phiDiffIndex(phi_diff)]++;
  // std::cout << "phidiffarr was incremented \n";
  theta_diff_arr[thetaDiffIndex(theta_diff)]++;
  // std::cout << "thetaDiffarr was incremented \n";
  theta_H_arr[thetaHalfIndex(theta_H)]++;
  // std::cout << "thetaHalfarr was incremented to " <<
  // theta_H_arr[thetaHalfIndex(H_local_spherical[0])] <<"\n";

  // Find index.
  // Note that phi_half is ignored, since isotropic BRDFs are assumed
  size_t red = ind;
  size_t green = ind + static_cast<size_t>(k_sampling_res_theta_h *
                                           k_sampling_res_theta_d *
                                           k_sampling_res_phi_d / 2);
  size_t blue =
      ind + static_cast<size_t>(k_sampling_res_theta_h *
                                k_sampling_res_theta_d * k_sampling_res_phi_d);

  brdf_proposal[red].push_back(px_value[0]);
  brdf_proposal[green].push_back(px_value[1]);
  brdf_proposal[blue].push_back(px_value[2]);
}

// filters out the top and bot filtersize% and averages the rest of the values,
// use value between 0 to 1
void MerlBaseBrdf::filter(const float filtersize) {
  int idx = 0;
  float filled = 0;

  // std::fstream bfile
  // =std::fstream("/home/stratman/Documents/code_public/light_stage_test/visualizebrdfs/indices.binary",
  // std::ios::out | std::ios::binary);

  for (auto row : brdf_proposal) {

    float size = static_cast<float>(row.size());
    // bfile.write((char *)&size, sizeof(float));

    if (size < 10) {
      // std::cout <<"brdf " << idx << " is empty ! \n";

      // DEBUG MARK AS RED when < 1
      if (size < 1) {
        // red
        brdf_[idx] = 0; 
                           // green
        // brdf_[idx + static_cast<size_t>(k_sampling_res_theta_h *
        //                                 k_sampling_res_theta_d *
        //                                 k_sampling_res_phi_d / 2)] = 0;
        // blue
        // brdf_[idx + static_cast<size_t>(k_sampling_res_theta_h *
        //                                 k_sampling_res_theta_d *
        //                                 k_sampling_res_phi_d)] = 0;

      } else {
        brdf_[idx] = std::accumulate(row.begin(), row.end(), 0) / size;
        filled++;
      }

      idx++;
      continue;
    }

    // sorts the row in ascending order by size
    std::sort(row.begin(), row.end());
    float avg, sum;

    size = size * filtersize;

    //+ size and -size ignore the bottom and top filtersize% of the row in the
    // calculations sums up all elements in the row and then averages them
    sum = std::accumulate(row.begin() + (size / 2.f), row.end() - (size / 2.f),
                          0);
    //  if (sum != 0){
    //  std::cout << "sum equals= " << std::to_string(sum) << "\n";
    //  }
    avg = sum / (static_cast<float>(row.size()) - size);

    // avg is clamped to a min of 0 to avoid negative numbers

    std::clamp(avg, 0.0f, avg);

    brdf_[idx] = static_cast<double>(avg);
    idx++, filled++;
  }

  // Linear Interpolate to smooth the brdf
  //  for (std::size_t t = 0; t < 100 ; ++t) {
  //     for (std::size_t i = 0; i + 1 < idx; ++i) {
  //             brdf_[i] = std::lerp(brdf_[i], brdf_[i + 1], .5);
  //         }}
  // bfile.close();
  std::fstream theta_H =
      std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                   "visualizebrdfs/theta_H.binary",
                   std::ios::out | std::ios::binary);
  std::fstream theta_diff =
      std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                   "visualizebrdfs/theta_diff.binary",
                   std::ios::out | std::ios::binary);
  std::fstream phi_diff =
      std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                   "visualizebrdfs/phi_diff.binary",
                   std::ios::out | std::ios::binary);
  std::fstream indices =
      std::fstream("/home/stratman/Documents/code_public/light_stage_test/"
                   "visualizebrdfs/indices_hemi.binary",
                   std::ios::out | std::ios::binary);

  theta_H.write((char *)&theta_H_arr, sizeof(int) * k_sampling_res_theta_h);
  theta_diff.write((char *)&theta_diff_arr,
                   sizeof(int) * k_sampling_res_theta_d);
  phi_diff.write((char *)&phi_diff_arr,
                 sizeof(int) * (k_sampling_res_phi_d / 2));
  indices.write((char *)&indices_arr, sizeof(int) * (k_sampling_res_phi_d / 2) *
                                          k_sampling_res_theta_d *
                                          k_sampling_res_theta_h);

  phi_diff.close();
  theta_diff.close();
  theta_H.close();
  indices.close();

  std::cout << "Brdf has been filled: "
            << (filled / static_cast<float>(idx)) * 100.f << " % " << filled
            << " out of " << idx << " have been filled \n";
}
// Lookup thetaHalf Index
int MerlBaseBrdf::thetaHalfIndex(float theta_half) const {
  if (theta_half <= 0.f) {
    return 0;
  }

  int temp =
      static_cast<int>(sqrt(theta_half * static_cast<float>((2.0 / M_PI))) *
                       static_cast<float>(k_sampling_res_theta_h));
  return std::clamp(temp, 0, k_sampling_res_theta_h - 1);
}

// Lookup thetaDiff Index
int MerlBaseBrdf::thetaDiffIndex(float theta_diff) const {

  int temp = static_cast<int>(theta_diff *
                              (2.f / static_cast<float>(M_PI) *
                               static_cast<float>(k_sampling_res_theta_d)));

  // std::cout << "theta diff index = " << temp << "\n";

  return std::clamp(temp, 0, k_sampling_res_theta_d - 1);
}
// Lookup phi_diff Index
int MerlBaseBrdf::phiDiffIndex(float phi_diff) const {
  // Because of reciprocity, the BRDF is unchanged under
  // phi_diff -> phi_diff + M_PI
  if (phi_diff < 0.f) {
    phi_diff += static_cast<float>(M_PI);
  }

  // In: phi_diff in [0 .. pi]
  // Out: tmp in [0 .. 179]
  return std::clamp(
      static_cast<int>(phi_diff *
                       (1.0f / static_cast<float>(M_PI) *
                        (static_cast<float>(k_sampling_res_phi_d) / 2.f))),
      0, (k_sampling_res_phi_d / 2) - 1);
}
