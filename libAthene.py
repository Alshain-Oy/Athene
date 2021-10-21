#!/usr/bin/env python3

import math
import cv2
import numpy as np
import mahotas

from sklearn.cluster import DBSCAN

class Utils:

    @staticmethod
    def auto_canny( image, sigma = 0.33 ):
        
        v = np.median( image )
        
        lower = int( max(0, (1.0 - sigma) * v) )
        upper = int( min(255, (1.0 + sigma) * v) )
        edged = cv2.Canny( image, lower, upper )
        
        return edged

    @staticmethod
    def merge_contours( contours ):
        out = contours[0].copy()
        for cnt in contours[1:]:
            out = np.append(out, cnt, axis = 0)        
        return out
         
    @staticmethod
    def preprocess_image( img, **kwargs ):
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        if kwargs.get( "useUnsharpMask", False):
            sg = kwargs.get( "UnsharpMaskSigma", 2.0 )
            blurred = cv2.GaussianBlur( img_bw, (0, 0), sg )
            img_bw = cv2.addWeighted( img_bw, 1.5, blurred, -0.5, 0, img_bw )

        scale = kwargs.get("imageScale", 1.0)
        bw = cv2.resize( img_bw, (0,0), fx = scale, fy = scale )
        #bw = img_bw
        edges = None

        if kwargs.get( "useCLAHE", False):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            bw = clahe.apply( bw )
        
        if kwargs.get( "histEq", False):
            bw = cv2.equalizeHist(bw)

        if kwargs.get( "useFiltering", True):
            bw = cv2.bilateralFilter(bw, 9, 17, 17)
            
            gs = kwargs.get("gaussianBlurSize", 3)
            bw = cv2.GaussianBlur(bw,(gs,gs),cv2.BORDER_DEFAULT)
            
        if kwargs.get("useOtsu", False):
            ret, bw = cv2.threshold( bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

        if kwargs.get("useCanny", True):
            edges = Utils.auto_canny( bw )
        else:
            edges = bw

        if kwargs.get("useAdaptiveTh", False):
            edges = cv2.adaptiveThreshold( bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, kwargs.get("adaptiveTileSize", 51), kwargs.get("adaptiveC", 6) )
        
        D = math.sqrt( edges.shape[0]**2 + edges.shape[1]**2)
        padding = int( (D - min(edges.shape)) / 2 + 1 )
        
        if kwargs.get( "addPadding", False ):
            edges = cv2.copyMakeBorder( edges, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0)
        else:
            padding = 0
        

        if kwargs.get("useClosing", True):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output = []

        for cnt in cnts:
            area = cv2.contourArea( cnt )
            if area < kwargs.get("contourMinSize", 25):
                continue
            
            if cnt.shape[0] < kwargs.get("contourMinLength", 5):
                continue

            output.append( cnt )
        
        return {"image": bw, "edges": edges, "contours": output, "all points": Utils.merge_contours(output), "padding": padding }    


    @staticmethod
    def compute_zernike_moments( image, contours ):
        out = {}
        sizes = np.zeros( (len(contours), )  )
        centres = {}

        for i in range( len(contours) ):
            
            cnt = contours[i]
            (cx,cy),radius = cv2.minEnclosingCircle(cnt)
            centres[i] = (cx,cy)
            
            x = int(cx - radius)
            y = int(cy - radius)
            w = int(2*radius)
            h = int(2*radius)

            plate = np.zeros( image.shape, dtype = np.uint8 )
            cv2.drawContours(plate, contours, i, 255, 3)
            
            plate = plate[ y-10 : y+h+10, x-10 : x+w+10 ]

            out[i] = mahotas.features.zernike_moments( plate, int(radius), degree=10 )
            ws = np.zeros(out[i].shape)

            # Weights for Zernike moments to red
            for n in range( len( ws ) ):
                ws[n] = math.pow(1.1, -n)
            
            out[i] = out[i] * ws

            sizes[i] = radius

        return {"ZM": out, "size": np.array( sizes ), "centres": centres}


    @staticmethod
    def corr_rotation_polar( plateA, plateB, **kwargs ):
        
        resize = kwargs.get( "rotationSearchResize", 1)
        minSize = kwargs.get( "minimumRotationImageSize", 36 )
        searchRange = kwargs.get( "rotationOffsetRange", 4)
        searchStep = kwargs.get( "rotationOffsetStepSize", 2)
        
        minShapeA = min(plateA.shape)
        
        if  minShapeA * resize < minSize:
            resize = minSize / minShapeA
        
        
        plateA = cv2.resize(plateA, (0,0), fx = resize, fy = resize)
        plateB = cv2.resize(plateB, (0,0), fx = resize, fy = resize)



        
        
        plateA = cv2.copyMakeBorder(plateA, 25, 25, 25, 25, cv2.BORDER_CONSTANT, 0 )

        rows, cols = plateB.shape
        centreBpix = np.array((cols/2, rows/2))
        radiusB = max([rows, cols])/2
        
        M = cv2.moments(plateB)
        mcxB = M["m10"] / M["m00"]
        mcyB = M["m01"] / M["m00"]
        centreB = np.array((mcxB, mcyB))


        rows, cols = plateA.shape
        radiusA = max([rows, cols])/2

        M = cv2.moments(plateA)
        mcxA = M["m10"] / M["m00"]
        mcyA = M["m01"] / M["m00"]
        centreA = np.array((mcxA, mcyA))

        radius = min([radiusA, radiusB])
        polarA = cv2.warpPolar( plateA, (0,0), centreA, radius, cv2.WARP_FILL_OUTLIERS )
        polarSearch = np.vstack([polarA, polarA])
        polarSearch = cv2.copyMakeBorder(polarSearch, 0, 0, 0, 15, cv2.BORDER_CONSTANT, 0)

        
        max_corr = 0
        max_angle = 0
        offset = [0, 0]

        for dx in range( -searchRange, searchRange + 1, searchStep):
            for dy in range( -searchRange, searchRange + 1, searchStep):

                polarB = cv2.warpPolar( plateB, (0,0), centreB + np.array([dx, dy]), radius - 1, cv2.WARP_FILL_OUTLIERS )
                
                res = cv2.matchTemplate(polarSearch, polarB, cv2.TM_CCOEFF_NORMED )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                angle = max_loc[1]/polarSearch.shape[0] * 720

                
                if max_val > max_corr:
                    max_corr = max_val
                    max_angle = angle
                    offset = np.array((-dx, -dy))


        offset[0] = offset[0]  - (centreB[0] - centreBpix[0])
        offset[1] = offset[1]  - (centreB[1] - centreBpix[1])
        
        
        return -max_angle, offset / resize



    @staticmethod
    def corr_rotation( plateA, plateB, **kwargs ):
        
        min_angle_delta = kwargs.get( "rotationPrecision", 0.5 )
        N = kwargs.get( "rotationSearchSegments", 7 )
        resize = kwargs.get( "rotationSearchResize", 0.35)
        minSize = kwargs.get( "minimumRotationImageSize", 36 )

        minShapeA = min(plateA.shape)
        if  minShapeA * resize < minSize:
            resize = minSize / minShapeA

        plateA = cv2.resize(plateA, (0,0), fx = resize, fy = resize)
        plateB = cv2.resize(plateB, (0,0), fx = resize, fy = resize)
        
        rows, cols = plateB.shape
        centre = (cols/2, rows/2)
        
        minError = -1
        minAngle = 0

        angle_start = 0
        angle_stop = 360  - 360 / N

        offset = None

        centre_offset =  np.array(plateA.shape)/2 - np.array(plateB.shape)/2


        while (angle_stop - angle_start) > min_angle_delta:
            
            minError = -1
            for i in range( N ):
                q = i / ( N - 1 )
                angle = angle_start * ( 1 - q ) + angle_stop * q

                M = cv2.getRotationMatrix2D( centre, angle, 1 )
                tplate = cv2.warpAffine( plateB, M, (cols, rows) )

                res = cv2.matchTemplate( tplate, plateA, cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                if max_val > minError:
                    minError = max_val
                    minAngle = angle
                    offset = max_loc + centre_offset
            
            delta = (angle_stop - angle_start) / N

            angle_start = minAngle - delta/2
            angle_stop = minAngle + delta/2
     
        return minAngle, offset / resize


    @staticmethod
    def bruteforce_contour_orientation( imageTemplate, imageContour, cntA, cntB, cntA_all, cntB_all, **kwargs ):
        
        x,y,w,h = cv2.boundingRect(cntA_all)
        plateA = imageTemplate

        x,y,w,h = cv2.boundingRect(cntB_all)
        border = 15
        plateB = cv2.copyMakeBorder(imageContour, 15, 15, 15, 15, cv2.BORDER_CONSTANT, 0)[y-border+15:y+h+border+15, x-border+15:x+w+border+15]
        

        H = max([plateA.shape[0], plateB.shape[0]])
        W = max([plateA.shape[1], plateB.shape[1]])

        
        padding = max([W, H])
        plateB = cv2.copyMakeBorder( plateB, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0)
        
        
        if kwargs.get( "useCartesianRotationSearch", False):
            minAngle, offset = Utils.corr_rotation( plateA, plateB, **kwargs)
        else:
            minAngle, offset = Utils.corr_rotation_polar( plateA, plateB, **kwargs)
        
        return minAngle, offset

    @staticmethod
    def compute_total_area( all_points ):
        rect = cv2.minAreaRect( all_points )
        return rect[1][0] * rect[1][1]


class ShapeMatcher( object ):
    def __init__( self, **kwargs ):
        self.templates = {}

        self.parameters = {}
        self.parameters.update( kwargs )

        self._clustering = DBSCAN( eps = self.parameters.get("cluster_eps", 20), min_samples = 3 )

    def configure( self, **kwargs ):
        self.parameters.update( kwargs )

        if "cluster_eps" in kwargs:
            self._clustering = DBSCAN( eps = kwargs.get( "cluster_eps" ), min_samples = 3 )


    def add_template( self, name, image, **kwargs ):
        args = {}
        args.update( self.parameters )
        args.update( kwargs )
        
        args["addPadding"] = True
        self.templates[ name ] = Utils.preprocess_image( image, **args )

        fingerprints = Utils.compute_zernike_moments( self.templates[ name ]["edges"], self.templates[ name ][ "contours" ] )
        self.templates[ name ].update( fingerprints )
        
        self.templates[ name ][ "area" ] = Utils.compute_total_area( self.templates[ name ][ "all points" ] ) 
    

    def detect( self, template_name, image, **kwargs ):
        args = {}
        args.update( self.parameters )
        args.update( kwargs )

        data = Utils.preprocess_image( image, **args )
        fingerprints = Utils.compute_zernike_moments( data["edges"], data[ "contours" ] )
        data.update( fingerprints )

        scaleFactor = 1.0 / args.get("imageScale", 1.0)

        template = self.templates[ template_name ]
        
        results = np.zeros( (len(template["ZM"]), len(data["ZM"]) ) )


        min_rel_size = 1 - args.get("size_range", 0.2)
        max_rel_size = 1 + args.get("size_range", 0.2)
        
        found_patterns = []
        matching_cnts = {}

        for i in template["ZM"].keys():
            for j in data["ZM"].keys():
                results[i][j] = 0
                size_ratio = template["size"][i] / data["size"][j]
                if size_ratio < min_rel_size or size_ratio > max_rel_size:
                    results[i][j] += 1
                else:
                    results[i][j] += np.linalg.norm( template["ZM"][i] - data["ZM"][j] )
                
                if j not in matching_cnts:
                    matching_cnts[j] = (None, 1e99)
                
                if matching_cnts[j][1] > results[i][j]:
                    matching_cnts[j] = (i, results[i][j])
        
        mins = np.min(results, axis = 0)

        if args.get("drawResults", True):
            outimg = image.copy()


        threshold = args.get( "threshold", 0.65 )
        
        selected_cnts = np.where( mins < threshold )[0]
        
        points = None
        indices = np.array([], dtype=np.int64)
        
        for i in selected_cnts:
            cnt = data["contours"][i]
            if args.get("drawResults", True):
                outimg = cv2.drawContours( outimg, [np.int0(cnt * scaleFactor)], -1, (0,255,0), 1 )
          
            if points is None:
                points = cnt.copy()
            else:
                points = np.append( points, cnt, axis = 0)
            
            indices = np.append( indices, np.ones( (len(cnt),) )*i, axis = 0)

        points = np.reshape( points, (-1,2) )
        
        cres = self._clustering.fit( points )
        labels = cres.labels_
        labels_unique = np.unique( labels )
        n_clusters = len( labels_unique )

        
        s = [None] * n_clusters
        for i in range( n_clusters ):
            d, = np.where(labels == i )
            s[i] = np.int0( np.unique(indices[d]) )

            if len(s[i]) < args.get("minClusterSize", 2):
                continue 
                
            
            scores = -data["size"][ s[i] ]
            s[i] = s[i][ np.int0( np.argsort(scores) ) ]

            cnts_cnt = []

            for cint in s[i]:
                cntB = data["contours"][ cint ]
                cnts_cnt.append( cntB )

            t_cnt = Utils.merge_contours(cnts_cnt)

            angle, offset = Utils.bruteforce_contour_orientation(template["image"], data["image"], template["contours"], cnts_cnt, template["all points"], t_cnt, **args )
            
            rect_cnt = cv2.minAreaRect(np.array(t_cnt))

            # area
            cnt_area = rect_cnt[1][0] * rect_cnt[1][1]
            relative_area = cnt_area / template["area"]

            
            if relative_area < args.get("minimumSpan", 0.15):
                continue

            
            rect_bbox = cv2.boundingRect( np.float32(t_cnt * scaleFactor) )

            match_x = -offset[1] * scaleFactor + rect_bbox[0] + rect_bbox[2] / 2
            match_y = -offset[0] * scaleFactor + rect_bbox[1] + rect_bbox[3] / 2

            if args.get("drawResults", True):
                
                cv2.line( outimg, np.int0((match_x, match_y-10)), np.int0((match_x, match_y+10)), (255,0,0), 1)
                cv2.line( outimg, np.int0((match_x-10, match_y)), np.int0((match_x+10, match_y)), (255,0,0), 1)
                
                dx = math.cos( angle * math.pi/180 )
                dy = math.sin( angle * math.pi/180 )
                
                nx = -dy
                ny = dx
                
                w = template["image"].shape[1] * scaleFactor / 2
                h = template["image"].shape[0] * scaleFactor / 2

                box = []
                box.append( [  match_x + w*dx + h * nx, match_y + w*dy + h * ny  ] )
                box.append( [  match_x + w*dx - h * nx, match_y + w*dy - h * ny  ] )
                box.append( [  match_x - w*dx - h * nx, match_y - w*dy - h * ny  ] )
                box.append( [  match_x - w*dx + h * nx, match_y - w*dy + h * ny  ] )


                box = np.int0(box)
                outimg = cv2.drawContours(outimg,[box],0,(0,0,255),2)


                arrow_len = 40
                arrow_tip_size = 5
                arrow = []

                arrow.append([  match_x, match_y ])
                arrow.append([  match_x + arrow_len*dx, match_y + arrow_len*dy ])

                arrow.append([  match_x + (arrow_len - arrow_tip_size)*dx + arrow_tip_size*nx, match_y + (arrow_len - arrow_tip_size)*dy + arrow_tip_size*ny ])
                arrow.append([  match_x + (arrow_len - arrow_tip_size)*dx - arrow_tip_size*nx, match_y + (arrow_len - arrow_tip_size)*dy - arrow_tip_size*ny ])
                arrow.append([  match_x + arrow_len*dx, match_y + arrow_len*dy ])

                arrow = np.int0(arrow)
                outimg = cv2.drawContours(outimg,[arrow],0,(255,0,0),2)
            
            pattern = {"angle": -angle, "position": (match_x, match_y) }
            found_patterns.append( pattern )
    
        if args.get( "drawResults", True):
            return found_patterns, outimg
        else:
            return found_patterns


