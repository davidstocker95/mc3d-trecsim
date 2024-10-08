#ifndef GMM_PARAM_H
#define GMM_PARAM_H

#include "config.h"
#include "mc3d_common.h"
#include <vector>
#include <tuple>
#include <iostream>

namespace MC3D_TRECSIM
{   
    template <typename Scalar>
    struct RelationMatrices {
        RowMatrix<int> connectionMatrix;      // Use int for binary relationship (0 or 1).
        RowMatrix<Scalar> meanMatrix;         // Use Scalar type for mean.
        RowMatrix<Scalar> varianceMatrix;     // Use Scalar type for variance.
    };

    template <typename Scalar>
    class GMMParam
    {
    public:
        std::vector<int> KEYPOINTS;
        std::vector<std::tuple<int, int, double, double>> LIMBS;
        Scalar limbRegulationFactor;
        Scalar nu;
        Scalar trackingIdBiasWeight;
        unsigned int maxIter;
        Scalar keypointConfidenceThreshold;
        Scalar tol;
        Scalar splineDegree;
        Scalar splineKnotDelta;
        unsigned int maxFrameBuffer;
        bool autoManageTheta;
        bool autoManageHypothesis;
        bool copyLastThetas;
        Scalar splineSmoothingFactor;
        size_t numSupportCameras;
        size_t notSupportedSinceThreshold;
        size_t responsibilityLookback;
        Scalar responsibilitySupportThreshold;
        Scalar totalResponsibilitySupportThreshold;
        bool dragAlongUnsupportedKeyPoints;
        unsigned int seed;
        int minValidKeyPoints;

        GMMParam() : KEYPOINTS(std::vector<int>()),
                     LIMBS(std::vector<std::tuple<int, int, double, double>>()),
                     limbRegulationFactor(0.0),
                     nu(1.0),
                     trackingIdBiasWeight(0.0),
                     maxIter(100),
                     keypointConfidenceThreshold(0.5),
                     tol(1.0),
                     splineDegree(3),
                     splineKnotDelta(500),
                     maxFrameBuffer(25),
                     autoManageTheta(false),
                     autoManageHypothesis(false),
                     copyLastThetas(false),
                     splineSmoothingFactor(0.0),
                     numSupportCameras(2),
                     notSupportedSinceThreshold(5),
                     responsibilityLookback(10),
                     responsibilitySupportThreshold(0.3),
                     totalResponsibilitySupportThreshold(0.3),
                     dragAlongUnsupportedKeyPoints(true),
                     seed(static_cast<unsigned>(time(0))),
                     minValidKeyPoints(0)
        {
            setSeed(seed);
        }

        unsigned int getSeed()
        {
            return seed;
        }

        void setSeed(unsigned int seed)
        {
            this->seed = seed;
            srand(seed);
        }

        // RelationMatrices getKeypointRelationMatrix()
        // {   
        //     RelationMatrices matrices;
        //     int numKeypoints = KEYPOINTS.size();
        //     matrices.connectionMatrix = RowMatrix<int>(numKeypoints, numKeypoints);
        //     matrices.meanMatrix = RowMatrix<Scalar>(numKeypoints, numKeypoints);
        //     matrices.varianceMatrix = RowMatrix<Scalar>(numKeypoints, numKeypoints);

        //     // Initialize matrices with default values
        //     matrices.connectionMatrix.setZero();
        //     matrices.meanMatrix.setZero();
        //     matrices.varianceMatrix.setZero();

        //     for (auto limb : LIMBS) {
        //         int keypoint1 = std::get<0>(limb);
        //         int keypoint2 = std::get<1>(limb);
        //         Scalar mean = std::get<2>(limb);
        //         Scalar variance = std::get<3>(limb);

        //         // Set connection to 1 (indicating a connection exists)
        //         matrices.connectionMatrix.coeffRef(keypoint1, keypoint2) = 1;
        //         matrices.connectionMatrix.coeffRef(keypoint2, keypoint1) = 1;

        //         // Set mean and variance values
        //         matrices.meanMatrix.coeffRef(keypoint1, keypoint2) = mean;
        //         matrices.meanMatrix.coeffRef(keypoint2, keypoint1) = mean;

        //         matrices.varianceMatrix.coeffRef(keypoint1, keypoint2) = variance;
        //         matrices.varianceMatrix.coeffRef(keypoint2, keypoint1) = variance;
        //     }

        //     return matrices;
        // }

        friend std::ostream &operator<<(std::ostream &os, const GMMParam &obj)
        {
            os << "GMMParams:" << std::endl;

            // Printing KEYPOINTS
            os << "\tKEYPOINTS: ";
            for (size_t i = 0; i < obj.KEYPOINTS.size(); ++i)
            {
                os << obj.KEYPOINTS[i];
                if (i < obj.KEYPOINTS.size() - 1) // Avoid trailing comma for the last element
                {
                    os << ", ";
                }
            }
            os << std::endl;

            // Printing LIMBS
            os << "\tLIMBS: " << std::endl;
            for (size_t i = 0; i < obj.LIMBS.size(); ++i)
            {
                os << "\t\tkeypoints: (" << std::get<0>(obj.LIMBS[i]) << ", " << std::get<1>(obj.LIMBS[i]) << "), "
                << "mu [%]: " << std::get<2>(obj.LIMBS[i]) << ", "
                << "std [%]: " << std::get<3>(obj.LIMBS[i]) << std::endl;
            }

            // Printing other members
            os << "\t" << "limbRegulationFactor: " << obj.limbRegulationFactor << std::endl;
            os << "\t" << "nu: " << obj.nu << std::endl;
            os << "\t" << "trackingIdBiasWeight: " << obj.trackingIdBiasWeight << std::endl;
            os << "\t" << "maxIter: " << obj.maxIter << std::endl;
            os << "\t" << "keypointConfidenceThreshold: " << obj.keypointConfidenceThreshold << std::endl;
            os << "\t" << "tol: " << obj.tol << std::endl;
            os << "\t" << "splineDegree: " << obj.splineDegree << std::endl;
            os << "\t" << "splineKnotDelta: " << obj.splineKnotDelta << std::endl;
            os << "\t" << "maxFrameBuffer: " << obj.maxFrameBuffer << std::endl;
            os << "\t" << "autoManageTheta: " << obj.autoManageTheta << std::endl;
            os << "\t" << "autoManageHypothesis: " << obj.autoManageHypothesis << std::endl;
            os << "\t" << "copyLastThetas: " << obj.copyLastThetas << std::endl;
            os << "\t" << "splineSmoothingFactor: " << obj.splineSmoothingFactor << std::endl;
            os << "\t" << "numSupportCameras: " << obj.numSupportCameras << std::endl;
            os << "\t" << "notSupportedSinceThreshold: " << obj.notSupportedSinceThreshold << std::endl;
            os << "\t" << "responsibilityLookback: " << obj.responsibilityLookback << std::endl;
            os << "\t" << "responsibilitySupportThreshold: " << obj.responsibilitySupportThreshold << std::endl;
            os << "\t" << "totalResponsibilitySupportThreshold: " << obj.totalResponsibilitySupportThreshold << std::endl;
            os << "\t" << "dragAlongUnsupportedKeyPoints: " << obj.dragAlongUnsupportedKeyPoints << std::endl;
            os << "\t" << "minValidKeyPoints: " << obj.minValidKeyPoints << std::endl;

            return os;
        }
    };
}
#endif