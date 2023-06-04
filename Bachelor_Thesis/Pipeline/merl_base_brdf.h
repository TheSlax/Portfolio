#ifndef MERL_BRDF_H
#define MERL_BRDF_H

#include "brdf.h"
#include "Vector.hh"
#include <vector>

class MerlBaseBrdf : public BRDF {
    public:
        /** Creates one of the BRDFs in the MERL database.
         * @param path The full path and name of the brdf-file.
         */
        MerlBaseBrdf(const std::string &path);
        ~MerlBaseBrdf();

        MerlBaseBrdf();

        base::Vec3f Evaluate(const base::Vec3f& w_i, const base::Vec3f& w_o);
        void RecordBrdf(const base::Vec3f& w_i, const base::Vec3f& w_o,
            const base::Vec3f& px_value, const base::Vec3f& norm, const base::Vec3f& tangent, const base::Vec3f& bitangent);
        base::Vec3f DebugBrdf(const base::Vec3f &w_i, const base::Vec3f &w_o, const base::Vec3f& normal);

        void filter(const float filtersize);
        void save(const std::string& path);

    private:
        double *brdf_;
        std::vector<std::vector<float>> brdf_proposal;  // all the data
        bool valid_;         // is brdf data valid?

        // converts two directions (in cartesian coordinates) to the needed
        // halfangle / difference angle coordinates.
        // Returns: (theta_half, phi_half, theta_diff, phi_diff)
        base::Vec4f stdCoordsToHalfDiffCoords(const base::Vec3f &w_i,
                const base::Vec3f &w_o, const base::Vec3f& norm, const base::Vec3f& bi_norm) const;

        // converts a normalized direction into the theta phi representation.
        base::Vec2f cartesian2sphereCoordNormalized(const base::Vec3f &d) const;
        // Rotates a vector around an axis
        base::Vec3f rotateVector(const base::Vec3f &v,
                const base::Vec3f &axis,
                float angle) const;
        // Lookup theta_half index
        // This is a non-linear mapping!
        // In:  [0 .. pi/2]
        // Out: [0 .. 89]
        int thetaHalfIndex(float theta_half) const;
        // Lookup theta_diff index
        // In:  [0 .. pi/2]
        // Out: [0 .. 89]
        int thetaDiffIndex(float theta_diff) const;
        int phiDiffIndex(float phi_diff) const;

       
        public:
        
        // Constants
        const int k_sampling_res_theta_h = 90; //90
        const int k_sampling_res_theta_d = 90; //90
        const int k_sampling_res_phi_d = 360;  //360
        const float k_red_scale = 1500.0f;
        const float k_green_scale = 1500.0f /1.15f;
        const float k_blue_scale = 1500.f;
        
        //debug arrays
        int phi_diff_arr[180] = {0};
        int theta_diff_arr[90] = { 0 };
        int theta_H_arr[90] = { 0 };
        int indices_arr[180 * 90 * 90] = { 0 };

};

#endif // MERL_BRDF_H
