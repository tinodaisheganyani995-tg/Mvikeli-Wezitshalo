/* =========================================================
   TOMATO DISEASE RECOGNITION SYSTEM
   MAIN JAVASCRIPT
   ========================================================= */


/* =========================================================
   1. MOBILE NAVBAR
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const menuToggle =
        document.getElementById("menu-toggle");

    const navLinks =
        document.getElementById("nav-links");


    /* -----------------------------------------------------
       OPEN / CLOSE MOBILE MENU
    ----------------------------------------------------- */

    if (menuToggle && navLinks) {

        menuToggle.addEventListener("click", function (event) {

            event.stopPropagation();

            navLinks.classList.toggle("active");

            const isOpen =
                navLinks.classList.contains("active");


            if (isOpen) {

                menuToggle.innerHTML =
                    '<i class="fa-solid fa-xmark"></i>';

                menuToggle.setAttribute(
                    "aria-label",
                    "Close menu"
                );

            } else {

                menuToggle.innerHTML =
                    '<i class="fa-solid fa-bars"></i>';

                menuToggle.setAttribute(
                    "aria-label",
                    "Open menu"
                );

            }

        });

    }


    /* -----------------------------------------------------
       CLOSE MENU AFTER CLICKING A LINK
    ----------------------------------------------------- */

    if (navLinks) {

        const links =
            navLinks.querySelectorAll("a");


        links.forEach(function (link) {

            link.addEventListener("click", function () {

                navLinks.classList.remove("active");


                if (menuToggle) {

                    menuToggle.innerHTML =
                        '<i class="fa-solid fa-bars"></i>';

                    menuToggle.setAttribute(
                        "aria-label",
                        "Open menu"
                    );

                }

            });

        });

    }


    /* -----------------------------------------------------
       CLOSE MENU WHEN CLICKING OUTSIDE
    ----------------------------------------------------- */

    document.addEventListener("click", function (event) {

        if (!menuToggle || !navLinks) {
            return;
        }


        const clickedInsideMenu =
            navLinks.contains(event.target);

        const clickedMenuButton =
            menuToggle.contains(event.target);


        if (
            !clickedInsideMenu &&
            !clickedMenuButton
        ) {

            navLinks.classList.remove("active");


            menuToggle.innerHTML =
                '<i class="fa-solid fa-bars"></i>';

            menuToggle.setAttribute(
                "aria-label",
                "Open menu"
            );

        }

    });


    /* -----------------------------------------------------
       CLOSE MENU WITH ESCAPE KEY
    ----------------------------------------------------- */

    document.addEventListener("keydown", function (event) {

        if (event.key !== "Escape") {
            return;
        }


        if (navLinks) {

            navLinks.classList.remove("active");

        }


        if (menuToggle) {

            menuToggle.innerHTML =
                '<i class="fa-solid fa-bars"></i>';

            menuToggle.setAttribute(
                "aria-label",
                "Open menu"
            );

        }

    });

});



/* =========================================================
   2. IMAGE SELECTION + CAMERA
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const cameraImage =
        document.getElementById("camera-image");

    const plantImage =
        document.getElementById("plant-image");

    const chooseImageButton =
        document.getElementById("choose-image-button");

    const imagePreview =
        document.getElementById("image-preview");

    const previewContainer =
        document.getElementById("selected-image-area");

    const predictionForm =
        document.getElementById("prediction-form");


    /* =====================================================
       CAMERA ELEMENTS
    ===================================================== */

    let cameraStream = null;

    let capturedCameraFile = null;


    /* =====================================================
       CREATE CAMERA WINDOW
    ===================================================== */

    const cameraModal = document.createElement("div");

    cameraModal.id = "camera-modal";

    cameraModal.innerHTML = `

        <div class="camera-box">

            <h3>
                <i class="fa-solid fa-camera"></i>
                Take Photo
            </h3>

            <video
                id="camera-video"
                autoplay
                playsinline>
            </video>

            <canvas
                id="camera-canvas"
                hidden>
            </canvas>

            <div class="camera-controls">

                <button
                    type="button"
                    id="capture-button"
                    class="camera-control-button">

                    <i class="fa-solid fa-camera"></i>
                    Capture

                </button>


                <button
                    type="button"
                    id="close-camera-button"
                    class="camera-control-button close-camera-button">

                    <i class="fa-solid fa-xmark"></i>
                    Cancel

                </button>

            </div>

        </div>

    `;


    /* Add modal to page */

    document.body.appendChild(cameraModal);


    const cameraVideo =
        document.getElementById("camera-video");

    const cameraCanvas =
        document.getElementById("camera-canvas");

    const captureButton =
        document.getElementById("capture-button");

    const closeCameraButton =
        document.getElementById("close-camera-button");


    /* =====================================================
       SHOW IMAGE PREVIEW
    ===================================================== */

    function showImagePreview(file) {

        if (!file) {
            return;
        }


        /* Check file type */

        if (!file.type.startsWith("image/")) {

            alert(
                "Please select an image file."
            );

            return;

        }


        /* Create temporary image URL */

        const imageURL =
            URL.createObjectURL(file);


        /* Display image */

        if (imagePreview) {

            imagePreview.src =
                imageURL;

        }


        /* Show preview */

        if (previewContainer) {

            previewContainer.classList.add(
                "active"
            );

        }

    }


    /* =====================================================
       CHOOSE IMAGE BUTTON
    ===================================================== */

    if (chooseImageButton && plantImage) {

        chooseImageButton.addEventListener(
            "click",
            function () {

                plantImage.click();

            }
        );

    }


    /* =====================================================
       CHOOSE IMAGE
    ===================================================== */

    if (plantImage) {

        plantImage.addEventListener(
            "change",
            function () {

                const file =
                    this.files[0];


                if (!file) {
                    return;
                }


                capturedCameraFile = null;

                showImagePreview(file);

            }
        );

    }


    /* =====================================================
       OPEN CAMERA
    ===================================================== */

    async function openCamera() {

        try {

            /* Request webcam permission */

            cameraStream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "environment"
                    },
                    audio: false
                });


            /* Connect webcam to video */

            cameraVideo.srcObject =
                cameraStream;


            /* Show camera */

            cameraModal.classList.add("active");

        }

        catch (error) {

            console.error(
                "Camera error:",
                error
            );


            alert(
                "Could not access the camera. Please allow camera permission in your browser."
            );

        }

    }


    /* =====================================================
       CLOSE CAMERA
    ===================================================== */

    function closeCamera() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(function (track) {

                    track.stop();

                });

            cameraStream = null;

        }


        cameraVideo.srcObject = null;

        cameraModal.classList.remove("active");

    }


    /* =====================================================
       TAKE PHOTO BUTTON
    ===================================================== */

    const takePhotoButton =
        document.getElementById(
            "take-photo-button"
        );


    if (takePhotoButton) {

        takePhotoButton.addEventListener(
            "click",
            function () {

                openCamera();

            }
        );

    }


    /* =====================================================
       CAPTURE PHOTO
    ===================================================== */

    if (captureButton) {

        captureButton.addEventListener(
            "click",
            function () {

                if (!cameraStream) {

                    return;

                }


                /* Set canvas size */

                cameraCanvas.width =
                    cameraVideo.videoWidth;

                cameraCanvas.height =
                    cameraVideo.videoHeight;


                /* Draw current camera frame */

                const context =
                    cameraCanvas.getContext("2d");


                context.drawImage(
                    cameraVideo,
                    0,
                    0,
                    cameraCanvas.width,
                    cameraCanvas.height
                );


                /* Convert image to JPEG */

                cameraCanvas.toBlob(
                    function (blob) {

                        if (!blob) {

                            alert(
                                "Could not capture the photo."
                            );

                            return;

                        }


                        /* Create a File object */

                        capturedCameraFile =
                            new File(
                                [blob],
                                "camera_photo.jpg",
                                {
                                    type: "image/jpeg"
                                }
                            );


                        /* Put captured image into
                           main Flask input */

                        if (plantImage) {

                            try {

                                const dataTransfer =
                                    new DataTransfer();


                                dataTransfer.items.add(
                                    capturedCameraFile
                                );


                                plantImage.files =
                                    dataTransfer.files;

                            }

                            catch (error) {

                                console.error(
                                    "Could not attach camera photo:",
                                    error
                                );

                            }

                        }


                        /* Show preview */

                        showImagePreview(
                            capturedCameraFile
                        );


                        /* Close camera */

                        closeCamera();

                    },
                    "image/jpeg",
                    0.95
                );

            }
        );

    }


    /* =====================================================
       CANCEL CAMERA
    ===================================================== */

    if (closeCameraButton) {

        closeCameraButton.addEventListener(
            "click",
            function () {

                closeCamera();

            }
        );

    }


    /* =====================================================
       CLOSE CAMERA WITH ESCAPE
    ===================================================== */

    document.addEventListener(
        "keydown",
        function (event) {

            if (
                event.key === "Escape" &&
                cameraModal.classList.contains("active")
            ) {

                closeCamera();

            }

        }
    );


    /* =====================================================
       CLOSE CAMERA BY CLICKING OUTSIDE
    ===================================================== */

    cameraModal.addEventListener(
        "click",
        function (event) {

            if (
                event.target === cameraModal
            ) {

                closeCamera();

            }

        }
    );

});



/* =========================================================
   3. PREDICTION FORM VALIDATION
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const predictionForm =
        document.getElementById(
            "prediction-form"
        );

    const plantImage =
        document.getElementById(
            "plant-image"
        );

    const cameraImage =
        document.getElementById(
            "camera-image"
        );

    const predictButton =
        document.getElementById(
            "predict-button"
        );


    /* Result page does not have this form */

    if (!predictionForm) {
        return;
    }


    predictionForm.addEventListener(
        "submit",
        function (event) {


            /* -------------------------------------------------
               CHECK IMAGE
            ------------------------------------------------- */

            let imageSelected =
                false;


            if (
                plantImage &&
                plantImage.files &&
                plantImage.files.length > 0
            ) {

                imageSelected =
                    true;

            }


            if (
                cameraImage &&
                cameraImage.files &&
                cameraImage.files.length > 0
            ) {

                imageSelected =
                    true;

            }


            /* -------------------------------------------------
               NO IMAGE
            ------------------------------------------------- */

            if (!imageSelected) {

                event.preventDefault();


                alert(
                    "Please take a photo or choose a tomato leaf image first."
                );


                return;

            }


            /* -------------------------------------------------
               GET SELECTED FILE
            ------------------------------------------------- */

            let selectedFile =
                null;


            if (
                plantImage &&
                plantImage.files &&
                plantImage.files.length > 0
            ) {

                selectedFile =
                    plantImage.files[0];

            }


            if (
                !selectedFile &&
                cameraImage &&
                cameraImage.files &&
                cameraImage.files.length > 0
            ) {

                selectedFile =
                    cameraImage.files[0];

            }


            /* -------------------------------------------------
               CHECK FILE TYPE
            ------------------------------------------------- */

            if (
                selectedFile &&
                !selectedFile.type.startsWith("image/")
            ) {

                event.preventDefault();


                alert(
                    "Please upload an image file."
                );


                return;

            }


            /* -------------------------------------------------
               DISABLE BUTTON
            ------------------------------------------------- */

            if (predictButton) {

                predictButton.disabled =
                    true;


                predictButton.innerHTML =
                    '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

            }

        }
    );

});



/* =========================================================
   4. CONFIDENCE PROGRESS BAR
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const progressBars =
        document.querySelectorAll(
            ".progress-bar"
        );


    if (!progressBars.length) {
        return;
    }


    progressBars.forEach(function (progressBar) {

        const width =
            parseFloat(
                progressBar.dataset.width
            );


        if (isNaN(width)) {
            return;
        }


        /* Keep value between 0 and 100 */

        const safeWidth =
            Math.max(
                0,
                Math.min(
                    100,
                    width
                )
            );


        /* Start at zero */

        progressBar.style.width =
            "0%";


        /* Animate */

        setTimeout(function () {

            progressBar.style.width =
                safeWidth + "%";

        }, 200);

    });

});



/* =========================================================
   5. RESULT IMAGE ERROR
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const resultImage =
        document.querySelector(
            ".result-image img"
        );


    if (!resultImage) {
        return;
    }


    resultImage.addEventListener(
        "error",
        function () {

            console.error(
                "The uploaded image could not be loaded."
            );

        }
    );

});



/* =========================================================
   6. SMOOTH SCROLL
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    const links =
        document.querySelectorAll(
            'a[href^="#"]'
        );


    links.forEach(function (link) {

        link.addEventListener(
            "click",
            function (event) {

                const targetID =
                    this.getAttribute(
                        "href"
                    );


                if (
                    !targetID ||
                    targetID === "#"
                ) {

                    return;

                }


                const target =
                    document.querySelector(
                        targetID
                    );


                if (target) {

                    event.preventDefault();


                    target.scrollIntoView({

                        behavior: "smooth",

                        block: "start"

                    });

                }

            }
        );

    });

});