"""
Creates different representations of point cloud.  Need to abstract server,
at the moment expects a "Earthmine" server object.
"""
import sys
import numpy as np
import vtk
import logging
from scipy import misc

# My libraries
from toolbox.progressbar import ProgressBar
from ipfe.misc import pixels_in, toECEF


class Atmosphere():
    """
    An atmosphere is set of different cloud representations
    """

    def __init__(self, width, height, location=((0, 0, 0)), at=((0, 1, 0)),
                 up=((1, 0, 0)), fov = 90, range_limit = 25, verbose=True):

        # point data (x,y,z,r,g,b)
        self.cloud = []

        # pixel and location data (lat,lon,alt,x,y)
        self.pix_loc = []

        # view model
        self.where = location  # absolute location in real space
        self.up = up  # incident with focal plane
        self.at = at  # orthogonal to focal plane
        self.fov = fov  # field of view

        # view constraints
        self.range_limit = range_limit  # cut off point for depth maps
        self.verbose = verbose  # provide progress bar and other information

        # The 'clouds' that make up the atmosphere (views of same cloud)
        self.image = np.zeros(shape=(width, height, 3), dtype='uint8')
        self.depth = np.zeros(shape=(width, height, 3), dtype='uint8')
        self.locations = np.zeros(shape=(width, height, 3), dtype='float32')
        self.polydata = vtk.vtkPolyData()

    def __to_world(self, x, y, point, width=512, height=512):
        """
        Adjust image(x,y) to correct location in space for a z
        """
        focal_length = width / 2.0
        xcenter = height / 2
        ycenter = width / 2
        point = point - self.where
        zr = np.dot(self.at, point)
        xr = zr * (xcenter - x) / focal_length
        yr = zr * (ycenter - y) / focal_length
        return xr, yr, zr

    def update(self, locations, pixels, image, vtkpts, cells, colours, n):
        """
        Populate a point cloud, at pixels, with locations and image

        Creates a modified image, a range image, and a VTK image (polydata).
        For the range image each pixel value represents the distance of a point
        to the provided focal plane.  Nearer points will be lighter than
        further away points. The range of grey values will be scaled between
        the camera location (view model) and the provided range limit (points
        equal to or further away than the provided range limit will be black).
        The VTK image and the modified RGB image will only contain values if a
        corresponding depth value exists.

        """
        for (c, r), location in zip(pixels, locations):
            (R, G, B) = image[r][c]
            try:

                # Convert lat/lon/alt to ECEF grid
                # KeyError indicates no location data
                point = toECEF(np.radians(location['lat']),
                              np.radians(location['lon']),
                              float(location['alt']))

                # have location data so proceede
                xr, yr, zr = self.__to_world(r, c, point, image.shape[0],
                                             image.shape[1])
                scaleFactor = abs(zr / self.range_limit)
                if scaleFactor > 1.0:
                    scaleFactor = 1.0
                depth = 255 - int(np.floor(scaleFactor * 255.0))
                self.depth[r][c] = (depth, depth, depth)
                self.image[r][c] = (R,G,B)
                self.locations[r][c] = (np.float32(locations['lat']),
                                        np.float32(location['lon']),
                                        np.float32(location['alt']))

                # Update VTK information
                if (zr < self.range_limit):  # don't add points to "far" away
                    vtkpts.InsertNextPoint(yr, xr, zr)
                    colours.InsertNextTuple3(R, G, B)
                    vert = vtk.vtkVertex()
                    vert.GetPointIds().SetId(0, n)
                    cells.InsertNextCell(vert)
                    n += 1  # count number of points added to cloud
                self.pix_loc.append((location['lat'], location['lon'],
                                     location['alt'], r, c, R, G, B))
            except:  # catch KeyError, no location data available
                (xr, yr, zr) = (0, 0, 0)

            self.cloud.append((xr, yr, zr, R, G, B))
        return n


    def fetch(self, server, image, roi=((0, 0, 1024, 1024))):
        """
        Locations are added and views updated as retreived from the server.
        """
        # vtk scafolding needed to incrementally updata vtkPolyData()
        vtkpts = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        colours = vtk.vtkUnsignedCharArray()
        colours.SetNumberOfComponents(3)
        colours.SetName("Colours")
        n = 0
        #print "ROI: ", roi
        numpixels = pixels_in(roi)
        if numpixels == 0:  # Do we have any pixels?
            logging.warning("Unable to fetch cloud. No pixels selected")
            return

        progress = ProgressBar("Getting locations...", numpixels)

        start = 0
        pixels = [(x, y) for x in range(roi[0], roi[2])
                  for y in range(roi[1], roi[3])]
        for end in range(500, numpixels, 500):
            locations = server.fetch_locations(pixels[start:end])
            n = self.update(locations, pixels[start:end], image, vtkpts,
                          cells, colours, n)
            start = end
            progress.update_display(end)
        if start <= len(pixels):
            locations = server.fetch_locations(pixels[start:len(pixels)])
            n = self.update(locations, pixels[start:len(pixels)],
                          image, vtkpts, cells, colours, n)
            progress.update_display(numpixels)

        # Update vtkPolyData
        self.polydata.SetPoints(vtkpts)
        self.polydata.SetVerts(cells)
        self.polydata.GetCellData().SetScalars(colours)
        self.polydata.Modified()

    def to_file(self, fileout):
        """
        Save as text file
        """
        of = open(fileout, 'w')
        fmtStr = "%15.4f %15.4f %15.4f\n"
        for (x, y, z) in (self.where, self.at, self.up):
            of.write(fmtStr % (float(x), float(y), float(z)))
        for points in self.cloud:
            of.write('%d   %d   %d   %d   %d   %d\n' % (points))
        of.close()

    def to_lla(self, fileout):
        """
        Save as text file
        """
        of = open(fileout, 'w')
        fmtStr = "%15.4f %15.4f %15.4f\n"
        for (x, y, z) in (self.where, self.at, self.up):
            of.write(fmtStr % (float(x), float(y), float(z)))
        for points in self.pix_loc:
            of.write('%15.7f   %15.7f   %15.7f   %d   %d   %d  %d  %d\n'
                     % (points))
        of.close()

    def to_vtp(self, filename):
        """
        Save points in VTK format so useable in VTK viewer
        """
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.polydata) #vtk5 SetInput()
        writer.SetDataModeToBinary()
        writer.Write()

    def save_depth(self, fileout):
        """
        Save the depth image
        """
        misc.imsave(fileout, self.depth)

    def save_image(self, fileout):
        """
        Save the modified RGB image
        """
        misc.imsave(fileout, self.image)
