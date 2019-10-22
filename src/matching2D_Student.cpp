#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCateory, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t = (double)cv::getTickCount();
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = ((descriptorCateory.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING: cv::NORM_L2);
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // student to finish
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        // student to finish
        int k = 2; 
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        const float ratio_thresh = 0.8;
        
        for(size_t i=0; i<knn_matches.size(); i++)
        {
            if(knn_matches[i][0].distance < ratio_thresh*knn_matches[i][1].distance)
                matches.push_back(knn_matches[i][0]);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << "knn matching in "  << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {

        // int nfeatures = 500;
        // float scaleFactor = 1.2;
        // int nlevels = 8;
        // int edgeThreshold=31;
        // int firstLevel=0;
        // int WTA_K=2;
        // cv::ORB::ScoreType scoreType= cv::ORB::HARRIS_SCORE;
        // int patchSize=31; 
        // int fastThreshold = 20;
        // extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        // bool orientationNormalized = true;
        // bool scaleNormalized = true;
        // float patternScale = 22.0f;
        // int octaves = 4;
        // const vector<int>& selectedPairs = vector<int>();
        // extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, octaves, selectedPairs);
        extractor = cv::xfeatures2d::FREAK::create();
    //    extractor->compute(img, keypoints, descriptors);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {

        // cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        // int descriptor_size = 0;
        // int descriptor_channels = 3;
        // float threshold = 0.001f;
        // int nOctaves = 4;
        // int nOctaveLayers = 4;
        // cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
        bool useProvidedKeypoints = false;
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {

        int nfeatures = 0;
        int nOctaves = 3;
        double contrastThreshold = 3;
        double edgeThreshold = 10;
        double sigma = 1.6;
        bool useProvidedKeypoints = false;
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaves, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return 1000*t/1.0;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return 1000*t/1.0;
}

// Self implementation:

double detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, const std::string detectorType, bool bVis)
{
    string windowName;
    double t = (double)cv::getTickCount();
    if (detectorType.compare("HARRIS") == 0)
    {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, true, k);

        // add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {

            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "Harris Corner Detector Results";
    }

    else if (detectorType.compare("FAST") == 0)
    {
        int threshold = 10;
        bool nonmaxSuppression = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
        fast->detect(img, keypoints, cv::noArray());
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "FAST Corner Detector Results";
    }

    else if (detectorType.compare("BRISK") == 0)
    {
        int threshold = 60;
        int octaves = 4;
        float patternScales = 1.0;
        cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(threshold, octaves, patternScales);
        bool useProvidedKeypoints = false;
        brisk->detect(img, keypoints, cv::noArray());
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "BRISK Corner Detector Results";
    }

    else if (detectorType.compare("ORB") == 0)
    {
        int nfeatures = 500;
        float scaleFactor = 1.2;
        int nlevels = 8;
        int edgeThreshold=31;
        int firstLevel=0;
        int WTA_K=2;
        cv::ORB::ScoreType scoreType= cv::ORB::HARRIS_SCORE;
        int patchSize=31; 
        int fastThreshold = 20;
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        orb->detect(img, keypoints, cv::noArray());
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "ORB Corner Detector Results";
    }

    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptor_size = 0;
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
        akaze->detect(img, keypoints, cv::noArray());
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "AKAZE Corner Detector Results";
    }

    else if (detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
        sift->detect(img, keypoints, cv::noArray());
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        windowName = "SIFT Corner Detector Results";
    }

    // visualize results
    bVis = false;
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    bVis = false;
    return 1000*t/1.0;
}
