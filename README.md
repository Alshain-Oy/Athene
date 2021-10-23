# Athene

Athene machine vision library

## Required libraries

- OpenCV >=4.5.x
- Numpy >= 1.21.x
- Mahotas >= 1.4.x
- Scikit-learn >= 0.24.x


## Athene shape matcher

- **Preprocessing**
  - Resizing image
  - Filtering and equalisation
  - Thresholding
  - Contour search
  - Prefiltering contours (discarding too small and large)

- **Pattern detection**
  - Compute Zernike moments for all contours
  - Select contours that are similar enough to the template
  - Find contours that are close to each other ( clustering )

- **Pattern matching** (for each cluster)
  - Test that cluster contains enough matching (unique) contours to the template
  - Test that total area of the cluster is comparable to the template
  - Compute orientation and centre of rotation compare to the template
  - Test that cross correlation between template and rotated pattern is high enough
  - Compute pattern position in the image

## List of parameters

| Name | Description | Stage | Default value |   |
| --- | --- | --- | --- | --- |
| threshold | Acceptance limit for contour similarity | Detection | 0.65 |  |
| minimumSpan | Minimum size of the pattern compared to template | Matching | 0.15 |  |
| minimumCorrelation | Minimum correlation for the pattern | Matching | 0.5 |  |
| contourMinArea | Minimum area for contours | Preprocessing | 25 | px^2  |
| contourMaxArea | Maximum area for contours | Preprocessing | image size / 4 | px^2  |
| contourMinLength | Minimum (arc) length for contours | Preprocessing | 5 | in px |
| contourMaxLength | Maximum (arc) length for contours | Preprocessing | 1e9 | in px |
| clusteringDistance | Maximum distance bewteen contours in cluster | Detection | 20 | px  |
| minClusterSize | Minimum amount of unique contours in a cluster | Matching | 2 |  |
| useUnsharpMask | Apply unsharp masking to images | Preprocessing | False |  |
| UnsharpMaskSigma | Sigma for unsharp masking Gaussian blur | Preprocessing | 2.0 |  |
| imageScale | Scale for resizing images before processing | Preprocessing | 1.0 |  |
| useCLAHE | Apply CLAHE to images | Preprocessing | False |  |
| useHistEq | Apply global histogram equalisation to images | Preprocessing | False |  |
| useBilateralFilter | Apply bilateral filtering to images | Preprocessing | True |  |
| useGaussianFilter | Apply Gaussian blur to images | Preprocessing | True |  |
| gaussianBlurSize | Kernel size for Gaussian blur | Preprocessing | 3 | in px  |
| useOtsu | Apply Otsu binarisation | Preprocessing | False |  |
| useAdaptiveTh | Apply adaptive thresholding to images | Preprocessing | False |  |
| invertAdaptive | Invert adaptive thresholding results | Preprocessing | False |  |
| adaptiveTileSize | Tile size for adaptive thresholding | Preprocessing | 51 | in px  |
| adaptiveC | Offset constant for adaptive thresholding | Preprocessing | 6 |  |
| useCanny | Apply Canny edge detection to images | Preprocessing | True |  |
| addPadding | Add padding to images | Preprocessing | False |  |
| useClosing | Apply closing operator to found edges | Preprocessing | True |  |
| maxZernikeRadius | Maximum radius for calculating Zernike moments | Detection | 16 | px |
| rotationSearchResize | Resize images before rotation search | Matching | 1.0 |  |
| minimumRotationImageSize | Minimum size for images in rotation search | Matching | 36 | N x N  |
| rotationOffsetRange | Range for searching centre of rotation (linear-polar) | Matching | 4 | px |
| rotationOffsetStepSize | Step size for centre of rotation search (linear-polar) | Matching | 2 | px |
| cartesianCorrelationTest | Perform second correlation test (linear-polar) | Matching | False |  |
| rotationPrecision | Precision of rotation search (cartesian) | Matching | 0.5 | deg |
| rotationSearchSegments | Rotation search segment size in angle space (cartesian) | Matching | 7 |  |
| useCartesianRotationSearch | Use cartesian rotation search instead of linear-polar | Matching | False |  |
| useEdgesForOrientation | Use all detected edges for orientation search | Matching | False |  |
| useContoursForOrientation | Use selected contours for orientation search | Matching | False |  |
| countourStippleFactor | Contour point reduction factor | Detection | 4 |  |
| drawResults | Render results in an image | All | True |  |
| stepByStep | Show progress step-by-step | All | False | Helps parameter selection |

