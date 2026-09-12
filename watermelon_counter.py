
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# ============================================================
# WATERMELON LONGITUDINAL STRIPE DETECTOR
#
# ONLY THE STRIPES FOLLOWING THE WATERMELON LENGTH ARE DETECTED
#
# Horizontal watermelon  -> horizontal stripes
# Vertical watermelon    -> vertical stripes
# Rotated watermelon     -> stripes follow its rotation
#
# GREEN = watermelon boundary
# BLUE  = detected watermelon stripes
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

MIN_WATERMELON_AREA = 5000

# Image filtering
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 60
BILATERAL_SIGMA_SPACE = 60

CLAHE_LIMIT = 2.0
CLAHE_GRID = 8

# Stripe settings
MIN_STRIPE_DISTANCE = 30

# Increase this if too many stripes are detected
# Decrease it if stripes are being missed
STRIPE_SENSITIVITY = 1.15


# ============================================================
# IMAGE FILTER
# ============================================================

def filter_image(image):

    # Reduce noise while preserving edges
    filtered = cv2.bilateralFilter(
        image,
        BILATERAL_DIAMETER,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE
    )

    # LAB gives better control over brightness
    lab = cv2.cvtColor(
        filtered,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_LIMIT,
        tileGridSize=(CLAHE_GRID, CLAHE_GRID)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# FIND WATERMELON
# ============================================================

def find_watermelon(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [20, 20, 15],
        dtype=np.uint8
    )

    upper_green = np.array(
        [105, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Clean mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest
    )

    if area < MIN_WATERMELON_AREA:
        return None, None

    watermelon_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        watermelon_mask,
        [largest],
        -1,
        255,
        -1
    )

    return largest, watermelon_mask


# ============================================================
# GET LENGTH, BREADTH AND ANGLE
# ============================================================

def get_dimensions(contour):

    rect = cv2.minAreaRect(
        contour
    )

    center = rect[0]
    width = rect[1][0]
    height = rect[1][1]
    angle = rect[2]

    # OpenCV angle convention is awkward.
    # We explicitly determine which dimension is LENGTH.

    if width >= height:

        length = width
        breadth = height

        # For width being the long side,
        # rect angle represents the long-axis angle.
        long_angle = angle

    else:

        length = height
        breadth = width

        # Height is the long side,
        # so add 90 degrees.
        long_angle = angle + 90.0

    # Normalize angle to -90 ... +90
    while long_angle >= 90:
        long_angle -= 180

    while long_angle < -90:
        long_angle += 180

    return (
        center,
        length,
        breadth,
        long_angle
    )


# ============================================================
# ROTATE WATERMELON SO LENGTH BECOMES HORIZONTAL
# ============================================================

def rotate_to_horizontal(
    image,
    mask,
    contour
):

    center, length, breadth, long_angle = (
        get_dimensions(contour)
    )

    center = (
        float(center[0]),
        float(center[1])
    )

    # Rotate the long axis onto horizontal.
    rotation_angle = -long_angle

    matrix = cv2.getRotationMatrix2D(
        center,
        rotation_angle,
        1.0
    )

    h, w = image.shape[:2]

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return (
        rotated_image,
        rotated_mask,
        length,
        breadth,
        long_angle,
        matrix
    )


# ============================================================
# FIND ONLY LONGITUDINAL STRIPES
#
# After rotation:
#
#        LENGTH
#   ------------------------>
#
#       ====================   stripe
#       ====================   stripe
#       ====================   stripe
#
# Therefore we look for stripe positions along the Y axis.
# ============================================================

def find_longitudinal_stripes(
    rotated_image,
    rotated_mask
):

    gray = cv2.cvtColor(
        rotated_image,
        cv2.COLOR_BGR2GRAY
    )

    # Strong blur removes small surface texture
    gray = cv2.GaussianBlur(
        gray,
        (15, 15),
        0
    )

    # Local background
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
    )

    # Dark areas become positive
    dark_response = (
        background.astype(np.float32)
        - gray.astype(np.float32)
    )

    # Outside watermelon = ignore
    dark_response[
        rotated_mask == 0
    ] = 0

    mask_float = (
        rotated_mask.astype(np.float32)
        / 255.0
    )

    dark_response = np.maximum(
        dark_response,
        0
    )

    weighted = (
        dark_response
        * mask_float
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Average across LENGTH.
    #
    # Each longitudinal stripe extends along the length,
    # so it produces a strong response at one Y position.
    # --------------------------------------------------------

    row_sum = np.sum(
        weighted,
        axis=1
    )

    row_count = np.sum(
        mask_float,
        axis=1
    )

    valid = row_count > 20

    profile = np.zeros_like(
        row_sum
    )

    profile[valid] = (
        row_sum[valid]
        / row_count[valid]
    )

    # Smooth the profile
    smooth_kernel = np.ones(
        31,
        dtype=np.float32
    ) / 31.0

    profile = np.convolve(
        profile,
        smooth_kernel,
        mode="same"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean_value = np.mean(
        profile[valid]
    ) if np.any(valid) else 0

    if mean_value <= 0:
        return [], profile

    threshold = (
        mean_value
        * STRIPE_SENSITIVITY
    )

    # --------------------------------------------------------
    # Find local peaks
    # --------------------------------------------------------

    candidates = []

    for y in range(
        2,
        len(profile) - 2
    ):

        if not valid[y]:
            continue

        if (
            profile[y] > profile[y - 1]
            and profile[y] >= profile[y + 1]
            and profile[y] > threshold
        ):

            candidates.append(
                (
                    y,
                    profile[y]
                )
            )

    # Strongest first
    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    selected = []

    for position, strength in candidates:

        too_close = False

        for old_position in selected:

            if abs(
                position - old_position
            ) < MIN_STRIPE_DISTANCE:

                too_close = True
                break

        if not too_close:

            selected.append(
                position
            )

    selected.sort()

    return selected, profile


# ============================================================
# DRAW STRIPE FOLLOWING WATERMELON LENGTH
# ============================================================

def draw_stripe(
    output,
    rotated_mask,
    y_position,
    inverse_matrix
):

    # Find the watermelon pixels at this row
    xs = np.where(
        rotated_mask[y_position, :] > 0
    )[0]

    if len(xs) < 30:
        return False

    x1 = int(
        np.min(xs)
    )

    x2 = int(
        np.max(xs)
    )

    # --------------------------------------------------------
    # Instead of one perfectly straight line, sample several
    # points and transform them back.
    #
    # This makes the result follow the actual orientation.
    # --------------------------------------------------------

    sample_x = np.linspace(
        x1,
        x2,
        100
    )

    points_original = []

    for x in sample_x:

        point = np.array(
            [
                [x, y_position]
            ],
            dtype=np.float32
        )

        transformed = cv2.transform(
            np.array([point]),
            inverse_matrix
        )[0]

        px = int(
            transformed[0][0]
        )

        py = int(
            transformed[0][1]
        )

        points_original.append(
            (px, py)
        )

    # Draw continuous stripe
    for i in range(
        len(points_original) - 1
    ):

        cv2.line(
            output,
            points_original[i],
            points_original[i + 1],
            (255, 0, 0),
            4
        )

    return True


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_watermelon(image):

    output = image.copy()

    # --------------------------------------------------------
    # Resize large image
    # --------------------------------------------------------

    max_width = 1200

    if output.shape[1] > max_width:

        scale = (
            max_width
            / output.shape[1]
        )

        output = cv2.resize(
            output,
            None,
            fx=scale,
            fy=scale
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    filtered = filter_image(
        output
    )

    # --------------------------------------------------------
    # FIND WATERMELON
    # --------------------------------------------------------

    contour, mask = find_watermelon(
        filtered
    )

    if contour is None:

        cv2.putText(
            output,
            "WATERMELON NOT DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        return (
            output,
            filtered,
            None
        )

    # --------------------------------------------------------
    # DRAW WATERMELON BOUNDARY
    # --------------------------------------------------------

    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # --------------------------------------------------------
    # GET DIMENSIONS
    # --------------------------------------------------------

    (
        center,
        length,
        breadth,
        angle
    ) = get_dimensions(
        contour
    )

    # --------------------------------------------------------
    # ROTATE WATERMELON
    # --------------------------------------------------------

    (
        rotated,
        rotated_mask,
        length,
        breadth,
        angle,
        rotation_matrix
    ) = rotate_to_horizontal(
        filtered,
        mask,
        contour
    )

    # --------------------------------------------------------
    # INVERSE ROTATION
    # --------------------------------------------------------

    inverse_matrix = cv2.invertAffineTransform(
        rotation_matrix
    )

    # --------------------------------------------------------
    # FIND ONLY LONGITUDINAL STRIPES
    # --------------------------------------------------------

    stripe_positions, profile = (
        find_longitudinal_stripes(
            rotated,
            rotated_mask
        )
    )

    # --------------------------------------------------------
    # DRAW BLUE STRIPES
    # --------------------------------------------------------

    actual_count = 0

    for y_position in stripe_positions:

        success = draw_stripe(
            output,
            rotated_mask,
            y_position,
            inverse_matrix
        )

        if success:

            actual_count += 1

    # --------------------------------------------------------
    # LENGTH / BREADTH RATIO
    # --------------------------------------------------------

    ratio = (
        length
        / max(breadth, 1)
    )

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        output,
        "LENGTH: "
        + str(int(length))
        + " px",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "BREADTH: "
        + str(int(breadth))
        + " px",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "LENGTH/BREADTH: "
        + str(round(ratio, 2)),
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "STRIPES: "
        + str(actual_count),
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        "STRIPE ANGLE: "
        + str(round(angle, 1))
        + " deg",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    # --------------------------------------------------------
    # DEBUG WINDOW
    #
    # Here the watermelon is normalized so its LENGTH is
    # horizontal. Only longitudinal stripes are shown.
    # --------------------------------------------------------

    debug = rotated.copy()

    cv2.drawContours(
        debug,
        cv2.findContours(
            rotated_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[0],
        -1,
        (0, 255, 0),
        2
    )

    for y_position in stripe_positions:

        cv2.line(
            debug,
            (0, y_position),
            (
                debug.shape[1],
                y_position
            ),
            (255, 0, 0),
            3
        )

    cv2.putText(
        debug,
        "LENGTH AXIS ->",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        debug,
        "STRIPES: "
        + str(actual_count),
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return (
        output,
        filtered,
        debug
    )


# ============================================================
# CAMERA MODE
# ============================================================

def camera_mode():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    print()
    print("CAMERA MODE")
    print("Place the watermelon on the platform.")
    print("Press Q to quit.")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera."
            )

            break

        # Mirror laptop camera
        frame = cv2.flip(
            frame,
            1
        )

        result, filtered, debug = (
            detect_watermelon(
                frame
            )
        )

        cv2.imshow(
            "1 - Watermelon Detection",
            result
        )

        cv2.imshow(
            "2 - Filtered Image",
            filtered
        )

        if debug is not None:

            cv2.imshow(
                "3 - Length Axis Analysis",
                debug
            )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if (
            key == ord("q")
            or key == ord("Q")
        ):

            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# IMAGE UPLOAD MODE
# ============================================================

def upload_mode():

    root = tk.Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Watermelon Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            (
                "JPG Files",
                "*.jpg *.jpeg"
            ),
            (
                "PNG Files",
                "*.png"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    root.destroy()

    if file_path == "":

        print(
            "No image selected."
        )

        return

    image = cv2.imread(
        file_path
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    result, filtered, debug = (
        detect_watermelon(
            image
        )
    )

    cv2.imshow(
        "1 - Watermelon Detection",
        result
    )

    cv2.imshow(
        "2 - Filtered Image",
        filtered
    )

    if debug is not None:

        cv2.imshow(
            "3 - Length Axis Analysis",
            debug
        )

    print()
    print("IMAGE PROCESSED")
    print("Press any key to close.")
    print()

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# MAIN MENU
# ============================================================

print()
print("================================================")
print("       WATERMELON STRIPE DETECTOR")
print("================================================")
print()
print("1. Laptop Camera")
print("2. Upload Image")
print("3. Exit")
print()

choice = input(
    "Enter your choice (1/2/3): "
)

if choice == "1":

    camera_mode()

elif choice == "2":

    upload_mode()

elif choice == "3":

    print(
        "Program closed."
    )

else:

    print(
        "Invalid choice."
    )