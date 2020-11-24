from typing import Union

from pyproj import Transformer, CRS
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import transform


def transform_coordinate_system(geoms, source_cs='EPSG:4326', target_cs='EPSG:3857'):
    project = Transformer.from_crs(CRS(source_cs), CRS(target_cs), always_xy=True).transform

    if isinstance(geoms, (list, tuple)):
        return [transform(project, geom) for geom in geoms]
    else:
        return transform(project, geoms)


def find_index_of_point_w_min_distance(list_of_coords, coord):
    temp = [Point(c).distance(Point(coord)) for c in list_of_coords]
    return temp.index(min(temp))


def cut_line_at_point(line, pt):
    # First coords of line
    if line.geom_type == 'LineString':
        coords = line.coords[:]
    else:
        coords = [geo.coords[:] for geo in line.geoms]
        coords = [item for sublist in coords for item in sublist]

    # Add the coords from the points
    coords += pt.coords
    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]
    # sort the coordinates
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    break_pt = find_index_of_point_w_min_distance(coords, pt.coords[:][0])

    #     break_pt = coords.index(pt.coords[:][0])
    if break_pt == 0:
        # it is the first point on the line, "line_before" is meaningless
        line_before = None
    else:
        line_before = LineString(coords[:break_pt + 1])
    if break_pt == len(coords) - 1:
        # it is the last point on the line, "line_after" is meaningless

        line_after = None
    else:
        line_after = LineString(coords[break_pt:])
    return line_before, line_after


def distance_btw_two_points_on_a_line(line: Union[LineString, MultiLineString], pt1: Point, pt2: Point):
    if line.geom_type == 'LineString':
        coords = line.coords[:]
    else:
        coords = [item for sublist in [geo.coords[:] for geo in line.geoms] for item in sublist]

    # Add the coords from the points
    coords += pt1.coords
    coords += pt2.coords
    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]
    # sort the coordinates
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    # get their orders
    first_pt = coords.index(pt1.coords[:][0])
    second_pt = coords.index(pt2.coords[:][0])
    if first_pt > second_pt:
        pt1, pt2 = pt2, pt1

    first_line_part = cut_line_at_point(line, pt1)[1]
    second_line_part = cut_line_at_point(first_line_part, pt2)[0]

    return second_line_part.length


def test_distance_btw_two_points_on_a_line():
    line = LineString([(0, 0), (180, 0)])
    pt1, pt2 = Point((0, 0)), Point((180, 0))

    print(distance_btw_two_points_on_a_line(*transform_coordinate_system([line, pt1, pt2])), 40075700 / 2)


if __name__ == '__main__':
    test_distance_btw_two_points_on_a_line()
