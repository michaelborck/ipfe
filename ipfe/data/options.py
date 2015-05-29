"""
Functions to process logic of command line of Atmosphere application
"""
import os
import sys
import logging
from argparse import ArgumentParser
from ConfigParser import ConfigParser
from geopy import geocoders
from data.earthmine_index import PanoInfo, PanoIndex, PanoGrid

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def get_options():
    """
    Reads in options from command line and reads config file.
    Command line overides the config file
    """
    arguments = parse_commandline()
    if arguments['configfile']:
        options = read_config(arguments['configfile'])

    # The commandline overides the config file
    if arguments['street']:
        arguments['pano_id'] = None
        options['pano_id'] = None

    opt_keys = options.keys()
    for key in arguments.keys():
        if (key not in opt_keys) or (key in opt_keys and arguments[key]):
            options[key] = arguments[key]

    options['lat'], options['lon'], options['alt'], options[
        'yaw'], pano_id = get_lat_lon_alt_yaw(options)
    options['image'] = str(pano_id) + ".jpg"
    options['points'] = str(pano_id) + ".raw"
    return options


def parse_commandline():
    """
    Parse the command line for various Earthmine options

    optional arguments:
      -h, --help            show this help message and exit
      --config FILE         file containing progrem settings.
      --cache FILE          directory where images and clouds are stored.
      --server IP:PORT      address of server
      --key KEY             public key for access
      --secret KEY          developer key for access
      --search_radius DEG   search radius
      --field_of_view DEG   field of view
      --roi NUM NUM NUM NUM
                        Define a rectangular (x0 y0 x1 y1) region of interest
      -d [FILE], --save_depth [FILE]
                        save depth map as a jpeg to FILE
      -sd, --show_depth     display depth image created
      -s3, --show_3D        display point cloud image created
      --all                 Generate all clouds, jpg, depth, vtp, txt.
      -a [FILE], --save_ascii [FILE]
                        save view model and points a ASCII to FILE
      -p [FILE], --save_polydata [FILE]
                        Save as vtkPolyData to FILE
      -j [FILE], --save_jpeg [FILE]
                        save RGB as a jpeg image to FILE
      -sj, --show_jpeg      display jpeg image created
      --index FILE          csv file to use to generate index. Assumes copy is
                        local \ to file system
      -xx, --no_index       Don't use local index file
      -t TILE, --tile TILE  given current position, select one of the four views
                        available: 'front', 'back', 'left', 'right'
      --width NUM           width of required image
      --height NUM          height of required image
      --panoid PANNO        panaramma id of an image
      --street ADDRESS      address of desired location
      -r, --random          fetch a random location
      --lla [DEG.MINS DEG.MINS DIST [DEG.MINS DEG.MINS DIST ...]],
      --lat_lon_alt [DEG.MINS DEG.MINS DIST [DEG.MINS DEG.MINS DIST..]]
                        use lat/lon/alt to obtain image and views
      -v, --verbose         display progress bars and other information
      -xc, --no_colour      exclude colour values for pixels without a depth value
      --range_limit NUM     range limit, 30-40 seems to work best
    """
    parser = ArgumentParser()
    parser.add_argument("--config", dest="configfile",
                        help="file containing progrem settings. Comandline will overide.",
                        metavar="FILE", default="getrgbd.cfg")
    parser.add_argument(
        "--server",
        dest="server",
        help="address of server",
        metavar="IP:PORT")
    parser.add_argument(
        "--key",
        dest="key",
        help="public key for access",
        metavar="KEY")
    parser.add_argument(
        "--secret",
        dest="secret",
        help="developer key for access",
        metavar="KEY")
    parser.add_argument(
        "--search_radius",
        dest="radius",
        help="search radius",
        metavar="DEG")
    parser.add_argument(
        "--field_of_view",
        dest="fov",
        help="field of view",
        metavar="DEG")
    parser.add_argument(
        "--roi", dest="roi", help="Define a rectangular region of interest",
        metavar="x0 y0 x1 y1", nargs=4, type=int)
    group_pts = parser.add_mutually_exclusive_group()
    # group_pts.add_argument("--point", dest="points",
                              # help="raw point file name. Defaults to
                              # 'PANO_ID.raw'", metavar="FILE")
    group_pts.add_argument(
        "-xp", "--no_points", action="store_false", dest="want_points",
        help="do not retrieve/save points to a file", default=True)
    group_img = parser.add_mutually_exclusive_group()
    group_img.add_argument("--image", dest="image",
                           help="image file name.  Defaults to 'PANO_ID.jpg'", metavar="FILE")
    group_img.add_argument(
        "-xi", "--no_image", action="store_false", dest="want_image",
        help="don't save retrieved image", default=True)
    group_indx = parser.add_mutually_exclusive_group()
    group_indx.add_argument("--index", dest="csvfile",
                            help="csv file to use to generate index. Assumes copy is local to file system",
                            metavar="FILE", default="AUS008_003.csv")
    group_indx.add_argument(
        "-xx", "--no_index", action="store_false", dest="use_local_index",
        help="Do not use local index file.  Rely on server", default=True)
    parser.add_argument("-t", "--tile", dest="tile",
                        help="given current position, select one of the four views available: \
                                  'front', 'back', 'left', 'right'  ",
                        choices=('front', 'back', 'left', 'right'), metavar="TILE", default='front')
    parser.add_argument(
        "--width",
        dest="width",
        help="width of required image",
        metavar="WIDTH",
        type=int)
    parser.add_argument(
        "--height",
        dest="height",
        help="height of required image",
        metavar="WIDTH",
        type=int)
    parser.add_argument(
        "--panoid",
        dest="pano_id",
        help="panaramma id of an image",
        metavar="PANNO",
        type=int)
    parser.add_argument(
        "--street",
        dest="street",
        help="address of desired location",
        metavar="ADDRESS")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="display progress bars and other information", default=False)
    parser.add_argument("-r", "--random", action="store_true", dest="random",
                        help="fetch a random location", default=False)
    parser.add_argument("--lat", dest="lat",
                        help="angular distance north or south of equator", metavar="DEG.MINS")
    parser.add_argument("--lon", dest="lon",
                        help="angular distance east or west of meridian", metavar="DEG.MINS", type=float)
    parser.add_argument(
        "--alt",
        dest="alt",
        help="altitude or height",
        metavar="DISTANCE",
        type=float)
    parser.add_argument("--yaw", dest="yaw",
                        help="twist about vertical axis", metavar="DEG.SEC", type=float)
    return vars(parser.parse_args())


def read_config(filename):
    ''' Default to local server
        Default location from From AUS3000.csv:
        panno_id   |     lat        |       lon         |   alt  |   yaw
        1000002630728 |115.87784527852 | -31.9528565703704 | 10.685 | 161.3977
    '''
       # config.set('Location', 'lon', "115.854240")
       # config.set('Location', 'lat', "-31.952671")
    config = ConfigParser()
    if os.path.exists(filename):
        config.read(filename)
    else:  # create an example.
        config.add_section('Earthmine')
        config.set('Earthmine', 'server', "127.0.0.1:11114")
        config.set('Earthmine', 'key', "j1gy260thvokxsde8ei1zfnq")
        config.set('Earthmine', 'secret', "UUSbuDoKpz")
        config.add_section('Location')
        config.set('Location', 'lat', "-31.95285657")
        config.set('Location', 'lon', "115.87784527")
        config.set('Location', 'alt', "10.685")
        config.set('Location', 'yaw', "161.3977")
        config.set('Location', 'street', "121 Hay Street, Perth")
        config.set('Location', 'pano_id', "1000002630728")
        config.add_section('Image')
        config.set('Image', 'tile', "front")
        config.set('Image', 'width', "512")
        config.set('Image', 'height', "512")
        config.set('Image', 'fov', "90")  # match tile on filesystem
        config.set('Image', 'radius', "80")
        with open(filename, 'wb') as configfile:
            config.write(configfile)
    # now format same as command line parser
    options = {}
    for section in config.sections():
        options = section_to_dict(config, section, options=options)
    return options


def get_lat_lon_alt_yaw(options):
    """"
    Only works for a local file which index the server connected to
    Get lat/lon/alt, No test for conflicting options.  For now assume
    user only specify one method, otherwise give preference as follows.
    Address in preference lat/lon/alt, pano_id in preference over address
    and random overides everything else.  Although the default lat/lon/alt
    is valid it may have been updated/changed on command line so need
    to ensure it is valid
    """
    pano_id = None  # Updated below, otherwise return None.
    lat = lon = alt = yaw = 0
    if options['use_local_index']:

        # Setup Index
        pano_info = PanoInfo()
        pano_info["name"] = options['csvfile']
        path = PanoIndex(pano_info)
        pano_grid = PanoGrid(pano_info)
        if options['random']:
            pano_id = path.random()
        elif options['pano_id']:
            try:
                pano_info[options['pano_id']]['lat']
                pano_id = options['pano_id']
            except:
                print "Unkown pano_id"
        elif options['street']:
            place, glat, glon = street_to_latlon(options['street'])
            pano_id = pano_grid.findClosest(glat, glon, 15)
        elif options['lat'] and options['lon']:
            try:
                pano_id = pano_grid.findClosest(
                    options['lat'],
                    options['lon'],
                    15)
            except:
                print "Unable to find anything close to lat/lon in databse"
        # Found a pano_id in local index file, lets get details of front tile
        if pano_id:
            lat = pano_info[pano_id]['lat']
            lon = pano_info[pano_id]['lon']
            alt = pano_info[pano_id]['alt']
            yaw = pano_info[pano_id]['yaw']

    # Still possible to do a reverse lookup to obtain lat/lon
    if not options['use_local_index'] and options['street']:
        pano_id, lat, lon = street_to_latlon(options['street'])

    return lat, lon, alt, yaw, pano_id


# Helper functions

def section_to_dict(config, section, options={}):
    """
    Converts a "section" from the cofig file to a dictionary
    """
    keys = config.options(section)
    for key in keys:
        try:
            options[key] = config.get(section, key)
            if options[key] == -1:
                logging.debug("skip: %s" % key)
        except:
            logging.debug("exception on %s!" % key)
            options[key] = None
    return options


def street_to_latlon(street):
    """
    Wrapper to enable reverse lookup
    """
    g = geocoders.GoogleV3()
    try:
        place, (glat, glon) = g.geocode(street)
    except:
        print "Unable to find lat/lon for address"
    return place, glat, glon
