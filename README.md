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

| Name | Description | Default value |   |
| --- | --- | --- | --- |
| threshold | Acceptance limit for contour similarity | 0.65 |  |
| minimumSpan | Minimum size of the pattern compared to template | 0.15 |  |
| minimumCorrelation | Minimum correlation for the pattern | 0.5 |  |
| contourMinArea | Minimum area for contours | 25 | px^2  |
| contourMaxArea | Maximum area for contours | image size / 4 | px^2  |
| contourMinLength | Minimum (arc) length for contours | 5 | in px |
| contourMaxLength | Maximum (arc) length for contours | 1e9 | in px |
| clusteringDistance | Maximum distance bewteen contours in cluster | 20 | px  |
| minClusterSize | Minimum amount of unique contours in a cluster | 2 |  |
| useUnsharpMask | Apply unsharp masking to images | False |  |
| UnsharpMaskSigma | Sigma for unsharp masking Gaussian blur | 2.0 |  |
| imageScale | Scale for resizing images before processing | 1.0 |  |
| useCLAHE | Apply CLAHE to images | False |  |
| useHistEq | Apply global histogram equalisation to images | False |  |
| useBilateralFilter | Apply bilateral filtering to images | True |  |
| useGaussianFilter | Apply Gaussian blur to images | True |  |
| gaussianBlurSize | Kernel size for Gaussian blur | 3 | in px  |
| useOtsu | Apply Otsu binarisation | False |  |
| useAdaptiveTh | Apply adaptive thresholding to images | False |  |
| invertAdaptive | Invert adaptive thresholding results | False |  |
| adaptiveTileSize | Tile size for adaptive thresholding | 51 | in px  |
| adaptiveC | Offset constant for adaptive thresholding | 6 |  |
| useCanny | Apply Canny edge detection to imges | True |  |
| addPadding | Add padding to images | False |  |
| useClosing | Apply closing operator to found edges | True |  |
| maxZernikeRadius | Maximum radius for calculating Zernike moments | 16 | px |
| rotationSearchResize | Resize images before rotation search | 1.0 |  |
| minimumRotationImageSize | Minimum size for images in rotation search | 36 | N x N  |
| rotationOffsetRange | Range for searching centre of rotation (linear-polar) | 4 | px |
| rotationOffsetStepSize | Step size for centre of rotation search (linear-polar) | 2 | px |
| cartesianCorrelationTest | Perform second correlation test (linear-polar) | False |  |
| rotationPrecision | Precision of rotation search (cartesian) | 0.5 | deg |
| rotationSearchSegments | Rotation search segment size in angle space (cartesian) | 7 |  |
| useCartesianRotationSearch | Use cartesian rotation search instead of linear-polar | False |  |
| useEdgesForOrientation | Use all detected edges for orientation search | False |  |
| useContoursForOrientation | Use selected contours for orientation search | False |  |
| countourStippleFactor | Contour point reduction factor | 4 |  |
| drawResults | Render results in an image | True |  |
| stepByStep | Show progress step-by-step | False | Helps parameter selection |

