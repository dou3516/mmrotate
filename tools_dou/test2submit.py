import json
import numpy as np
import math
from tqdm import tqdm


classes = ['Sovremenny-class destroyer', '052C-destroyer', 'Bunker', '636-hydrographic survey ship', '903A-replenishment ship', 'Tuzhong Class Salvage Tug', 'Traffic boat', '082II-Minesweeper', 'unknown', 'Emory S. Land-class submarine tender', 'Submarine', 'Barracks Ship', 'Whidbey Island-class dock landing ship', 'San Antonio-class amphibious transport dock', 'Arleigh Burke-class Destroyer', 'Ticonderoga-class cruiser', 'Barge', 'Sand Carrier', 'Oliver Hazard Perry-class frigate', 'Towing vessel', '022-missile boat', '037-submarine chaser', '904B-general stores issue ship', '072III-landing ship', '926-submarine support ship', 'Independence-class littoral combat ship', 'Avenger-class mine countermeasures ship', 'Mercy-class hospital ship', '052D-destroyer', '074-landing ship', '529-Minesweeper', 'USNS Bob Hope', '051-destroyer', 'Fishing Vessel', 'Freedom-class littoral combat ship', 'Nimitz Aircraft Carrier', 'Wasp-class amphibious assault ship', 'Sacramento-class fast combat support ship', 'Lewis and Clark-class dry cargo ship', '001-aircraft carrier', 'Xu Xiake barracks ship', 'Lewis B. Puller-class expeditionary mobile base ship', 'USNS Spearhead', '072A-landing ship', '081-Minesweeper', 'Takanami-class destroyer', '680-training ship', '920-hospital ship', '073-landing ship', 'Other Warship', '272-icebreaker', 'unknown auxiliary ship', '053H2G-frigate', '053H3-frigate', 'Container Ship', '053H1G-frigate', '903-replenishment ship', 'Yacht', 'Powhatan-class tugboat', 'YG-203 class yard gasoline oiler', 'YW-17 Class Yard Water', 'YO-25 class yard oiler', 'Asagiri-class Destroyer', 'Hiuchi-class auxiliary multi-purpose support ship', 'Henry J. Kaiser-class replenishment oiler', '072II-landing ship', '904-general stores issue ship', '056-corvette', '054A-frigate', '815-spy ship', '037II-missile boat', '037-hospital ship', '905-replenishment ship', '054-frigate', 'Abukuma-class destroyer escort', 'JMSDF LCU-2001 class utility landing crafts', 'Tenryu-class training support ship', 'Kurobe-class training support ship', 'Zumwalt-class destroyer', '071-amphibious transport dock', 'Tank ship', 'Iowa-class battle ship', 'Bulk carrier', 'Tarawa-class amphibious assault ship', '922A-Salvage lifeboat', 'Blue Ridge class command ship', '908-replenishment ship', '052B-destroyer', 'Hatsuyuki-class destroyer', 'Hatsushima-class minesweeper', 'Hyuga-class helicopter destroyer', 'Mashu-class replenishment oilers', 'Kongo-class destroyer', 'Towada-class replenishment oilers', 'Hatakaze-class destroyer', '891A-training ship', '721-transport boat', 'Akizuki-class destroyer', 'Osumi-class landing ship', 'Murasame-class destroyer', 'Uraga-class Minesweeper Tender', '909A-experimental ship', '074A-landing ship', '051C-destroyer', 'Hayabusa-class guided-missile patrol boats', '679-training ship', 'Forrestal-class Aircraft Carrier', 'Kitty Hawk class aircraft carrier', 'JS Suma', '909-experimental ship', 'Izumo-class helicopter destroyer', 'JS Chihaya', '639A-Hydroacoustic measuring ship', '815A-spy ship', 'North Transfer 990', 'Cyclone-class patrol ship', '052-destroyer', '917-lifeboat', '051B-destroyer', 'Yaeyama-class minesweeper', '635-hydrographic Survey Ship', 'USNS Montford Point', '925-Ocean salvage lifeboat', '648-submarine repair ship', '625C-Oceanographic Survey Ship', 'Sugashima-class minesweepers', 'Uwajima-class minesweepers', 'Northampton-class tug', 'Hibiki-class ocean surveillance ships', '055-destroyer', 'Futami-class hydro-graphic survey ships', 'JS Kurihama', '901-fast combat support ship']


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point_single(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point_single(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point_single(polys)
    return polys

def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def main(jsonfile, csvfile):

    with open(jsonfile, 'r') as f:
        json_info = json.load(f)
        f.close()

    #
    strs = []
    for info in tqdm(json_info):
        rbboxes = np.hstack((info['bbox'], info['score']))
        poly = obb2poly_np(rbboxes, version='le90')  # [x_ctr,y_ctr,w,h,angle,score]
        str = "%s.bmp,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.5f\n" \
              % (info['image_id'], classes[info['category_id']],
                 poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], poly[8])
        strs.append(str)

    with open(csvfile, 'w') as fw:
        fw.write('ImageID,LabelName,X1,Y1,X2,Y2,X3,Y3,X4,Y4,Conf\n')
        fw.writelines(strs)
        fw.close()


if __name__ == '__main__':
    cstr = 'shiprs133'
    jsonfile = './work_dirs/' + cstr + '.bbox.json'
    csvfile = './work_dirs/' + cstr + '.csv'
    main(jsonfile, csvfile)
