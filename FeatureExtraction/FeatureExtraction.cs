using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;

// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class Program
    {
        // file paths
        private const string inputFilePath = @"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\input.jpg";

        // The main program entry point
        static void Main(string[] args)
        {
            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\shape_predictor_68_face_landmarks.dat"))
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // draw the landmark points on the image
                    for (var i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                    }
                }

                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");
            }
        }
    }
}