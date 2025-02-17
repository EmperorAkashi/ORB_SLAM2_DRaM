
#include "RawORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

// Constructor with default parameters similar to ORB-SLAM2
RawORBMatcher::RawORBMatcher(int nFeatures=1000, 
                float scaleFactor=1.2f,
                int nLevels=8,
                int iniThFAST=20,
                int minThFAST=7,
                float nnRatio=0.6f,
                bool checkOrientation=true)
    : mfNNratio(nnRatio), mbCheckOrientation(checkOrientation)
{
    // Create ORB extractor
    mpORBextractor = new ORBextractor(nFeatures, scaleFactor, 
                                        nLevels, iniThFAST, minThFAST);
}

// Extract fresh features from a KeyFrame's image
void RawORBMatcher::ExtractFeatures(KeyFrame* pKF, 
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors) 
{
    // Get original image from KeyFrame
    cv::Mat im = pKF->imgLeft;  // or appropriate image getter
    
    // Extract new features
    cv::Mat mask;  // empty mask
    (*mpORBextractor)(im, mask, keypoints, descriptors);
}

// Match features between two sets of keypoints/descriptors
std::vector<cv::DMatch> RawORBMatcher::MatchFeatures(
    const std::vector<cv::KeyPoint>& kp1,
    const cv::Mat& desc1,
    const std::vector<cv::KeyPoint>& kp2,
    const cv::Mat& desc2)
{
    std::vector<cv::DMatch> matches;
    
    // For each keypoint in first set
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.knnMatch(desc1, desc2, knn_matches, 2);
    
    // Apply ratio test and optionally check orientation
    for(const auto& knn_match : knn_matches) {
        if(knn_match.size() < 2) continue;
        
        const cv::DMatch& m = knn_match[0];
        const cv::DMatch& n = knn_match[1];
        
        if(m.distance < mfNNratio * n.distance) {
            if(mbCheckOrientation) {
                float rot = kp1[m.queryIdx].angle - kp2[m.trainIdx].angle;
                if(rot < 0.0f)
                    rot += 360.0f;
                int bin = round(rot*factor);
                if(bin == HISTO_LENGTH)
                    bin = 0;
                assert(bin>=0 && bin<HISTO_LENGTH);
                rotHist[bin].push_back(m);
            } else {
                matches.push_back(m);
            }
        }
    }
    
    // If checking orientation, get matches from the dominant orientation
    if(mbCheckOrientation) {
        int max=0;
        for(int i=0; i<HISTO_LENGTH; i++) {
            if(rotHist[i].size() > max) {
                max = rotHist[i].size();
            }
        }
        
        // Add matches from bins with enough matches
        for(int i=0; i<HISTO_LENGTH; i++) {
            if(rotHist[i].size() > 0.1f*max)
                matches.insert(matches.end(), 
                                rotHist[i].begin(),
                                rotHist[i].end());
        }
    }
    
    return matches;
}

// Get 2D-3D correspondences from matches
void RawORBMatcher::Get2D3DCorrespondences(
    KeyFrame* pKF,
    const std::vector<cv::KeyPoint>& queryKPs,
    const std::vector<cv::DMatch>& matches,
    std::vector<cv::Point2f>& points2D,
    std::vector<cv::Point3f>& points3D)
{
    points2D.clear();
    points3D.clear();
    
    // Get MapPoints from KeyFrame
    const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPoints();
    const std::vector<cv::KeyPoint>& kfKeyPoints = pKF->mvKeysUn;
    
    for(const auto& match : matches) {
        const int kfIdx = match.trainIdx;  // Index in KeyFrame
        MapPoint* pMP = vpMapPoints[kfIdx];
        
        if(pMP) {
            // Get 3D point
            cv::Mat pos = pMP->GetWorldPos();
            points3D.push_back(cv::Point3f(pos.at<float>(0),
                                            pos.at<float>(1),
                                            pos.at<float>(2)));
                                            
            // Get 2D point from query keypoints
            points2D.push_back(queryKPs[match.queryIdx].pt);
        }
    }
}

} // namespace ORB_SLAM2