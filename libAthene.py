#!/usr/bin/env python3

import math
import cv2
import numpy as np
import mahotas
import time
from sklearn.base import MultiOutputMixin

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
            mode = cv2.THRESH_BINARY_INV
            if kwargs.get("invertAdaptive", False):
                mode = cv2.THRESH_BINARY
            edges = cv2.adaptiveThreshold( bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, kwargs.get("adaptiveTileSize", 51), kwargs.get("adaptiveC", 6) )
        
        D = math.sqrt( edges.shape[0]**2 + edges.shape[1]**2)
        padding = int( (D - min(edges.shape)) / 2 + 1 )
        
        if kwargs.get( "addPadding", False ):
            edges = cv2.copyMakeBorder( edges, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0)
        else:
            padding = 0
        

        if kwargs.get("useClosing", True):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      
        output = []

        for cnt in cnts:
            area = cv2.contourArea( cnt )
            if area < kwargs.get("contourMinArea", 25):
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
            t0_total = time.time()
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

            t0 = time.time()
            out[i] = mahotas.features.zernike_moments( plate, int(radius), degree=8 )
            t1 = time.time()
            #print( "mahotas took: %.2f ms, radius = %.1f, µs/pix = %.2f "%((t1-t0)*1000, radius, (t1-t0)*1e6/(radius*radius)))

            ws = np.zeros(out[i].shape)

            # Weights for Zernike moments to red
            for n in range( len( ws ) ):
                ws[n] = math.pow(1.1, -n)
            
            out[i] = out[i] * ws

            sizes[i] = radius
            t1_total = time.time()
            print( "Total zernike time: %.1f ms, moments: %.1f ms, all rest: %.1f ms"%(  (t1_total-t0_total)*1000, (t1-t0)*1000, (t1_total-t0_total - (t1-t0))*1000 ) )

        return {"ZM": out, "size": np.array( sizes ), "centres": centres}


    @staticmethod
    def corr_rotation_polar( plateA, plateB, plateBoriginalSize, **kwargs ):
        
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
        polarSearch = cv2.copyMakeBorder(polarSearch, 0, 0, 0, 2, cv2.BORDER_CONSTANT, 0)

        
        max_corr = 0
        max_angle = 0
        offset = [0, 0]

        best_polarB = None
        for dx in range( -searchRange, searchRange + 1, searchStep):
            for dy in range( -searchRange, searchRange + 1, searchStep):

                polarB = cv2.warpPolar( plateB, (0,0), centreB + np.array([dx, dy]), radius - 1, cv2.WARP_FILL_OUTLIERS )
                
                res = cv2.matchTemplate(polarSearch, polarB, cv2.TM_CCOEFF )
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                angle = max_loc[1] / polarSearch.shape[0] * 720

                
                if max_val > max_corr:
                    max_corr = max_val
                    max_angle = angle
                    offset = np.array((-dx, -dy))
                    best_polarB = polarB.copy()

        res = cv2.matchTemplate(polarSearch, best_polarB, cv2.TM_CCOEFF_NORMED )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        max_corr = max_val        
        
        if kwargs.get("secondaryCorrelationTest", True):
            M = cv2.getRotationMatrix2D( centreB, -max_angle, 1 )
            plateB_rot = cv2.warpAffine( plateB, M, (plateB.shape[1], plateB.shape[0]) )
            
            template = plateA.copy()
            
            if plateB_rot.shape[0] > plateA.shape[0] and plateB_rot.shape[1] > plateA.shape[1]:

                if plateBoriginalSize[0]*resize + 2*abs(offset[1]) < plateA.shape[0]:
                    if plateBoriginalSize[1]*resize + 2*abs(offset[0]) < plateA.shape[1]:
                        acx = plateA.shape[1] / 2 - offset[0]
                        acy = plateA.shape[0] / 2 - offset[1]
                        x0 = int( acx - (plateBoriginalSize[1]*resize-2) / 2)
                        x1 = int( acx + (plateBoriginalSize[1]*resize-2) / 2)
                        
                        y0 = int( acy - (plateBoriginalSize[0]*resize-2) / 2)
                        y1 = int( acy + (plateBoriginalSize[0]*resize-2) / 2)
                        template = template[y0:y1, x0:x1]

                res = cv2.matchTemplate(plateB_rot, template, cv2.TM_CCORR_NORMED )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                max_corr = max( max_val, max_corr)
            

            
        offset[0] = offset[0]  - (centreB[0] - centreBpix[0])
        offset[1] = offset[1]  - (centreB[1] - centreBpix[1])
        
        offset = np.array( offset )
        

        return -max_angle, (offset / resize), max_corr



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
     
        return minAngle, (offset / resize), minError


    @staticmethod
    def bruteforce_contour_orientation( imageTemplate, imageContour, cntA, cntB, cntA_all, cntB_all, **kwargs ):
        
        x,y,w,h = cv2.boundingRect(cntA_all)
        plateA = imageTemplate

        x,y,w,h = cv2.boundingRect(cntB_all)
        border = 15
        plateB = cv2.copyMakeBorder(imageContour, 15, 15, 15, 15, cv2.BORDER_CONSTANT, 0)[y-border+15:y+h+border+15, x-border+15:x+w+border+15]
        plateBoriginalSize = plateB.shape

        H = max([plateA.shape[0], plateB.shape[0]])
        W = max([plateA.shape[1], plateB.shape[1]])

        
        padding = max([W, H])//2
        plateB = cv2.copyMakeBorder( plateB, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0)
        
        t0 = time.time()
        if kwargs.get( "useCartesianRotationSearch", False):
            minAngle, offset, max_corr = Utils.corr_rotation( plateA, plateB, **kwargs)
        else:
            minAngle, offset, max_corr = Utils.corr_rotation_polar( plateA, plateB, plateBoriginalSize, **kwargs)
        t1 = time.time()
        print( "corr_rotation_polar took: %.1f ms"%((t1-t0)*1000))
        return minAngle, offset, max_corr

    @staticmethod
    def compute_total_area( all_points ):
        rect = cv2.minAreaRect( all_points )
        return rect[1][0] * rect[1][1]


    @staticmethod
    def draw_found_pattern( outimg, template, scaleFactor, match ):
        match_x, match_y = match["position"]
        angle = match["angle"]

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

        return outimg

    
    @staticmethod
    def detect_pattern( template, data, cluster, clusterIdx, scaleFactor, tmp_images, args ):
        scores = -data["size"][ cluster ]
        cluster = cluster[ np.int0( np.argsort(scores) ) ]

        cnts_cnt = []

        for cint in cluster:
            cntB = data["contours"][ cint ]
            cnts_cnt.append( cntB )

            if args.get( "stepByStep", False):
                cv2.drawContours( tmp_images["outimg_sbs"], np.int0([cntB * scaleFactor]), 0, (255,255,255), 3)
                cv2.drawContours( tmp_images["outimg_sbs"], np.int0([cntB * scaleFactor]), 0, (0,0,0), 1)
            
        


        t_cnt = Utils.merge_contours(cnts_cnt)
        rect_cnt = cv2.minAreaRect(np.array(t_cnt))
        
        # area
        cnt_area = rect_cnt[1][0] * rect_cnt[1][1]
        relative_area = cnt_area / template["area"]

        
        if relative_area < args.get("minimumSpan", 0.15):
            return None
        
        if args.get("useEdgesForOrientation", False):
            angle, offset, max_corr = Utils.bruteforce_contour_orientation(template["edges"], data["edges"], template["contours"], cnts_cnt, template["all points"], t_cnt, **args )
        
        elif args.get("useContoursForOrientation", False):
            angle, offset, max_corr = Utils.bruteforce_contour_orientation(template["drawn_contours"], tmp_images["drawn_contours_image"], template["contours"], cnts_cnt, template["all points"], t_cnt, **args )
        
        else:
            angle, offset, max_corr = Utils.bruteforce_contour_orientation(template["image"], data["image"], template["contours"], cnts_cnt, template["all points"], t_cnt, **args )
        
        
        if args.get( "stepByStep", False):
            tmp_img = tmp_images["outimg_sbs"].copy()
            outh, outw = tmp_images["outimg_sbs"].shape[:2]
            
            box = cv2.boxPoints(rect_cnt)
            box = np.int0(box*scaleFactor)
            cv2.drawContours(tmp_img,[box],0,(0,0,255),2)
            
            text_colour = (0,0,255)
            cv2.putText( tmp_img, "Correlation: %.2f" % max_corr, (outw - 230, outh - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 5, cv2.LINE_AA )
            cv2.putText( tmp_img, "Correlation: %.2f" % max_corr, (outw - 230, outh - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_colour, 1, cv2.LINE_AA )
            cv2.putText( tmp_img, "Cluster idx: %i" % clusterIdx, (outw - 230, outh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 5, cv2.LINE_AA )
            cv2.putText( tmp_img, "Cluster idx: %i" % clusterIdx, (outw - 230, outh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_colour, 1, cv2.LINE_AA )
            
            cv2.imshow( "Step-by-Step", tmp_img)
            cv2.waitKey(-1)


        if max_corr < args.get("minimumCorrelation", 0.5):
            return None
        
        rect_bbox = cv2.boundingRect( np.float32(t_cnt * scaleFactor) )

        match_x = -offset[1] * scaleFactor + rect_bbox[0] + rect_bbox[2] / 2
        match_y = -offset[0] * scaleFactor + rect_bbox[1] + rect_bbox[3] / 2
        
        pattern = {"angle": angle, "position": (match_x, match_y), "score": {"correlation": max_corr, "span": relative_area, "clusterSize": len(cluster)} }
        return pattern

    @staticmethod
    def detect_pattern_mp( args ):
        return Utils.detect_pattern( *args )


class ShapeMatcher( object ):
    def __init__( self, **kwargs ):
        self.templates = {}

        self.parameters = {}
        self.parameters.update( kwargs )

        self._clustering = DBSCAN( eps = self.parameters.get("clusteringDistance", 20), min_samples = 3 )


        self._timings = []
        self._t0 = 0

    def configure( self, **kwargs ):
        self.parameters.update( kwargs )

        if "clusteringDistance" in kwargs:
            self._clustering = DBSCAN( eps = kwargs.get( "clusteringDistance" ), min_samples = 3 )

    def _start_clock( self ):
        self._t0 = time.time()
    
    def _stop_clock( self, name ):
        self._timings.append( (name, (time.time() - self._t0)*1000 ) ) 

    def add_template( self, name, image, **kwargs ):
        args = {}
        args.update( self.parameters )
        args.update( kwargs )
        
        args["addPadding"] = True
        self.templates[ name ] = Utils.preprocess_image( image, **args )

        fingerprints = Utils.compute_zernike_moments( self.templates[ name ]["edges"], self.templates[ name ][ "contours" ] )
        self.templates[ name ].update( fingerprints )
        
        self.templates[ name ][ "area" ] = Utils.compute_total_area( self.templates[ name ][ "all points" ] ) 
    
    def get_template_specs( self, name ):
        out = {}
        out["name"] = name
        out["area"] = self.templates[name]["area"]
        out["Ncontours"] = len( self.templates[name]["contours"] )
        out["contours"] = []
        for cnt in self.templates[name]["contours"]:
            entry = {}
            entry["length"] = cnt.shape[0]
            entry["area"] = cv2.contourArea( cnt )
            out["contours"].append( entry )
        return out



    def detect( self, template_name, image, **kwargs ):
        args = {}
        args.update( self.parameters )
        args.update( kwargs )

        self._start_clock()
        data = Utils.preprocess_image( image, **args )
        self._stop_clock("preprocess")

        self._start_clock()
        fingerprints = Utils.compute_zernike_moments( data["edges"], data[ "contours" ] )
        data.update( fingerprints )
        self._stop_clock("fingerprinting")
        print( "len(data[contours]) =",len(data["contours"]) )

        scaleFactor = 1.0 / args.get("imageScale", 1.0)

        template = self.templates[ template_name ]
        
        results = np.zeros( (len(template["ZM"]), len(data["ZM"]) ) )


        min_rel_size = 1 - args.get("sizeRange", 0.2)
        max_rel_size = 1 + args.get("sizeRange", 0.2)
        
        found_patterns = []
        matching_cnts = {}
        
        self._start_clock()

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
        
        threshold = args.get( "threshold", 0.65 )
        
        selected_cnts = np.where( mins < threshold )[0]
        
        self._stop_clock("contour matching")

        temporary_images = {}
        if args.get("drawResults", True):
            outimg = image.copy()
        
        if args.get( "stepByStep", False):
            temporary_images["outimg_sbs"] = image.copy()
        

        
        points = None
        indices = np.array([], dtype=np.int64)
        
        self._start_clock()
        
        if args.get("useContoursForOrientation", False):
            temporary_images["drawn_contours_image"] = np.zeros( data["edges"].shape, dtype=np.uint8 )
            for i in selected_cnts:
                cnt = data["contours"][i]
                cv2.drawContours(temporary_images["drawn_contours_image"], [cnt], -1, 255, 2)
            
            if "drawn_contours" not in template:
                template["drawn_contours"] = np.zeros( template["edges"].shape, dtype=np.uint8 )
                cv2.drawContours(template["drawn_contours"], template["contours"], -1, 255, 2)


        self._stop_clock("temporary images")

        self._start_clock()

        for i in selected_cnts:
            cnt = data["contours"][i]
            if args.get("drawResults", True):
                outimg = cv2.drawContours( outimg, [np.int0(cnt * scaleFactor)], -1, (0,255,0), 1 )
          

            stipple_cnt = cnt[::args.get("countourStippleFactor", 4)]
            if points is None:
                points = stipple_cnt.copy()
            else:
                points = np.append( points, stipple_cnt, axis = 0)
            
            indices = np.append( indices, np.ones( (len(stipple_cnt),) )*i, axis = 0)

        self._stop_clock("clustering (preops)")

        self._start_clock()

        points = np.reshape( points, (-1,2) )
        
        cres = self._clustering.fit( points )
        labels = cres.labels_
        labels_unique = np.unique( labels )
        
        candidate_clusters = []

        for clusterIdx in labels_unique:
            d, = np.where(labels == clusterIdx )
            cluster = np.int0( np.unique(indices[d]) )

            if len(cluster) < args.get("minClusterSize", 2):
                continue 
            
            candidate_clusters.append( (clusterIdx, cluster))
        
        self._stop_clock("clustering (rest)")

        
        for (clusterIdx, cluster) in candidate_clusters:
            self._start_clock()
            pattern = Utils.detect_pattern( template, data, cluster, clusterIdx, scaleFactor, temporary_images, args )
            if pattern is not None:
                found_patterns.append( pattern )
            self._stop_clock("per-pattern-%s" % clusterIdx )


        if args.get( "drawResults", True):
            self._start_clock()
            for pattern in found_patterns:
                outimg = Utils.draw_found_pattern( outimg, template, scaleFactor, pattern )
            self._stop_clock("drawing results")          
            return found_patterns, outimg

        else:
            return found_patterns


