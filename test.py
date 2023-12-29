import json

def extract_rectangles(data, rectangles):
    for item in data:
        if isinstance(item, dict):
            properties = item.get("properties")
            if properties is not None:
                rectangle = properties.get("rectangle")
                if rectangle is not None:
                    rectangles.append(rectangle)

        elif isinstance(item, list):
            extract_rectangles(item, rectangles)
    # print(rectangles)
    # print(type(rectangles))
    rectangles=[(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles]
    print(rectangles)
    print(type(rectangles))

with open("./input/case7/lat.json", "r") as f:
    data = json.load(f)

rectangles = []
extract_rectangles(data.values(), rectangles)
# print(rectangles)
