#ifndef RAWORBMATCHER_H
#define RAWORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"

namespace ORB_SLAM2
{
    class RawORBmatcher
    {
    public:
            // Constructor with default parameters similar to ORB-SLAM2
    RawORBmatcher(int nFeatures=1000, 
                  float scaleFactor=1.2f,
                  int nLevels=8,
                  int iniThFAST=20,
                  int minThFAST=7,
                  float nnRatio=0.6f,
                  bool checkOrientation=true);
    
    // Extract fresh features from a KeyFrame's image
    void ExtractFeatures(KeyFrame* pKF, 
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors);


        // Match features between two sets of keypoints/descriptors
    std::vector<cv::DMatch> MatchFeatures(
        const std::vector<cv::KeyPoint>& kp1,
        const cv::Mat& desc1,
        const std::vector<cv::KeyPoint>& kp2,
        const cv::Mat& desc2);

    // Get 2D-3D correspondences from matches
    void Get2D3DCorrespondences(
        KeyFrame* pKF,
        const std::vector<cv::KeyPoint>& queryKPs,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point2f>& points2D,
        std::vector<cv::Point3f>& points3D);

    private:
    ORBextractor* mpORBextractor;
    float mfNNratio;
    bool mbCheckOrientation;
    
    // For orientation histogram
    static const int HISTO_LENGTH = 30;
    static const float factor = HISTO_LENGTH/360.0f;
    std::vector<cv::DMatch> rotHist[HISTO_LENGTH];      

    }

};

# endif // RAWORBMATCHER_H