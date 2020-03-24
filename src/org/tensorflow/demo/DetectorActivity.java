/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.GridView;
import android.widget.Toast;

// import com.opencsv.CSVWriter;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;
import java.util.Vector;

import org.json.JSONObject;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.
import android.provider.Settings.Secure;
import android.view.View.OnTouchListener;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  // private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb"; // "file:///android_asset/pothole-yolov3-tiny.pb"; // YingLH
  // private static final String YOLO_MODEL_FILE = "file:///android_asset/pothole-yolov3-tiny.pb"; // YingLH
  // private static final String YOLO_MODEL_FILE = "file:///android_asset/pothole-yolov2-voc.pb"; // YingLH
  // private static final String YOLO_MODEL_FILE = "file:///android_asset/pothole-yolov2-tiny-voc.pb"; // YingLH
  // private static final String YOLO_MODEL_FILE = "file:///android_asset/pothole-tiny-yolo-voc.pb"; // YingLH
  private static final String YOLO_MODEL_FILE = "file:///android_asset/pothole-tiny-yolo-voc-detect.pb"; // YingLH
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO
  }
  // private static final DetectorMode MODE = DetectorMode.TF_OD_API; //YOLO; // YingLH
  private static final DetectorMode MODE = DetectorMode.YOLO; // YingLH

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.1f; // 0.25f; // YingLH

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = true;
  private static final boolean SAVE_ORIGINAL_BITMAP = true;
  private static final boolean SAVE_DETECTION_RESULT_BITMAP = true;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;
  private double latitude = 0.0;
  private double longitude = 0.0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e(e, "Exception initializing classifier!");
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            // YingLH-key: 4. draw(): draw detected bounding boxes
            tracker.draw(canvas);
            if (isDebug()) {
              // YingLh-key: 4. drawDebug()
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            // YingLH Start
            GPSTracker gps = new GPSTracker(getApplicationContext());
            latitude = gps.getLatitude();
            longitude = gps.getLongitude();
            lines.add(String.format("lat:%.4f, lon:%.4f", latitude, longitude));
            // YingLH End
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            // YingLH-key: display rotation debugging text
            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines); // YingLH
          }
        });
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();

            // YingLH-key: 1. recognizeImage(): call the model to detect bounding boxes
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            String uuid = UUID.randomUUID().toString();

            Integer i = 0;
            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence
                      && result.getLocation().top > (cropCopyBitmap.getHeight() * 2.0 / 5.0)
                      && result.getLocation().bottom > (cropCopyBitmap.getHeight() * 1.0 / 10.0)
                      && result.getLocation().left > (cropCopyBitmap.getWidth() * 1.0 / 6.0)
                      && result.getLocation().right < (cropCopyBitmap.getWidth() * 5.0 / 6.0)
                    ) { // YingLH-Rule: ignore the result from the top 2/5 part of the image
                        // YingLH-Rule: ignore the result from the bottom 1/10 part of the image
                        // YingLH-Rule: ignore the result from the left and right 1/6 part of the image
                // YingLH-key: 2. drawRect() draw detection bounding box on debug image
                canvas.drawRect(location, paint); // YingLH
                canvas.drawText(String.format("%.2f", result.getConfidence()),
                        result.getLocation().right, result.getLocation().bottom, paint); // YingLH

                sendDetectionResult2Server(uuid + "-" + (i++).toString(),
                        Math.round(result.getLocation().left),
                        Math.round(result.getLocation().top),
                        Math.round(result.getLocation().width()),
                        Math.round(result.getLocation().height()),
                        result.getConfidence());

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            // YingLH Start
            rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);


            if (SAVE_ORIGINAL_BITMAP == true) {
              ImageUtils.saveBitmap(croppedBitmap, uuid + "_original.jpg");
            }

            if (SAVE_DETECTION_RESULT_BITMAP == true) {
              ImageUtils.saveBitmap(cropCopyBitmap, uuid + "_detection_result.jpg");
            }

            // Save to photo gallery in JPEG
            //MediaStore.Images.Media.insertImage(getContentResolver(), croppedBitmap,
            //        uuid + "_original", "description");
            //MediaStore.Images.Media.insertImage(getContentResolver(), cropCopyBitmap,
            //        uuid + "_detection_result", "description");
            // YingLH End

            // YingLH-key: 3. trackResults(): draw bounding boxes on the debugging images after pressing volume key
            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });
  }

  /*
  private void writeToCSV() {
    String baseDir = android.os.Environment.getExternalStorageDirectory().getAbsolutePath();
    String fileName = "AnalysisData.csv";
    String filePath = baseDir + File.separator + fileName;
    File f = new File(filePath);
    CSVWriter writer;
    FileWriter mFileWriter;

    // File exist
    if(f.exists()&&!f.isDirectory())
    {
      mFileWriter = new FileWriter(filePath, true);
      writer = new CSVWriter(mFileWriter);
    }
    else
    {
      writer = new CSVWriter(new FileWriter(filePath));
    }

    String[] data = {"Ship Name", "Scientist Name", "..."};

    writer.writeNext(data);

    writer.close();
  }
   */

  private void sendDetectionResult2Server(final String uuid,
                        final int boundingbox_x, final int boundingbox_y,
                        final int boundingbox_width, final int boundingbox_height,
                        final float confidence) {
    Thread thread = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          URL url = new URL(getString(R.string.ApiURL));
          HttpURLConnection conn = (HttpURLConnection) url.openConnection();
          conn.setRequestMethod("POST");
          conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
          conn.setRequestProperty("Accept","application/json");
          conn.setRequestProperty("x-api-key", getString(R.string.ApiKey));
          conn.setDoOutput(true);
          conn.setDoInput(true);

          JSONObject jsonParam = new JSONObject();
          jsonParam.put("uuid", uuid);
          jsonParam.put("boundingbox_x", boundingbox_x);
          jsonParam.put("boundingbox_y", boundingbox_y);
          jsonParam.put("boundingbox_width", boundingbox_width);
          jsonParam.put("boundingbox_height", boundingbox_height);
          jsonParam.put("confidence", confidence);
          jsonParam.put("latitude", latitude);
          jsonParam.put("longitude", longitude);
          jsonParam.put("device_id", Settings.Secure.getString(getContentResolver(),
                  Settings.Secure.ANDROID_ID));

          Log.i("JSON", jsonParam.toString());
          DataOutputStream os = new DataOutputStream(conn.getOutputStream());
          //os.writeBytes(URLEncoder.encode(jsonParam.toString(), "UTF-8"));
          os.writeBytes(jsonParam.toString());

          os.flush();
          os.close();

          Log.i("STATUS", String.valueOf(conn.getResponseCode()));
          Log.i("MSG" , conn.getResponseMessage());

          conn.disconnect();
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    });

    thread.start();
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }
}
