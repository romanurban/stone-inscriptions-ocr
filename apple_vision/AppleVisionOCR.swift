import Vision
import ImageIO
import Foundation

// Function to perform text detection and return detected text
func detectText(in imagePath: URL, relativeTo basePath: URL) -> [String: String] {
    var result: [String: String] = ["filename": "", "detected": ""]
    
    guard let imageSource = CGImageSourceCreateWithURL(imagePath as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        print("Unable to create image from path: \(imagePath.path)")
        return result
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let request = VNRecognizeTextRequest { request, error in
        guard error == nil else {
            print("Text detection error: \(error!.localizedDescription)")
            return
        }
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            print("No text detected.")
            return
        }

        result["detected"] = observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
        print("Detected text in \(imagePath.lastPathComponent): \(result["detected"]!)")
    }

    do {
        try handler.perform([request])
    } catch {
        print("Failed to perform text detection request: \(error.localizedDescription)")
    }

    // Set the relative filename in the result
    let relativePath = imagePath.path.replacingOccurrences(of: basePath.path, with: "")
    result["filename"] = relativePath

    return result
}

// Function to recursively process all images in directories and subdirectories
func processImagesInDirectory(at url: URL, basePath: URL, results: inout [[String: String]]) {
    let fileManager = FileManager.default
    guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles], errorHandler: nil) else {
        print("Failed to create directory enumerator for: \(url.path)")
        return
    }

    for case let fileURL as URL in enumerator {
        let fileAttributes = try? fileURL.resourceValues(forKeys: [.isDirectoryKey])
        if let isDirectory = fileAttributes?.isDirectory, isDirectory {
            print("Entering directory: \(fileURL.path.replacingOccurrences(of: basePath.path, with: ""))")
            continue
        } else if ["png", "jpg", "jpeg"].contains(fileURL.pathExtension.lowercased()) {
            results.append(detectText(in: fileURL, relativeTo: basePath))
        }
    }
}

// Function to save OCR results to a JSON file
func saveResultsToJSON(_ results: [[String: String]], in directoryPath: URL) {
    let timestamp = Int(Date().timeIntervalSince1970)
    let filename = "ocr_result_\(timestamp).json"
    let resultsDirectory = directoryPath.appendingPathComponent("results", isDirectory: true)
    let resultsURL = resultsDirectory.appendingPathComponent(filename)

    do {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: resultsDirectory, withIntermediateDirectories: true, attributes: nil)
        var data = try JSONSerialization.data(withJSONObject: results, options: .prettyPrinted)
        
        // Convert to String and replace escaped slashes
        if let jsonString = String(data: data, encoding: .utf8) {
            let unescapedJsonString = jsonString.replacingOccurrences(of: "\\/", with: "/")
            data = Data(unescapedJsonString.utf8)
        }
        
        try data.write(to: resultsURL)
        print("Results saved to \(resultsURL.path)")
    } catch {
        print("Failed to save results to JSON: \(error.localizedDescription)")
    }
}


// Example usage
let userName = "username"
let directoryPath = URL(fileURLWithPath: "/Users/\(userName)/dataset")
var results = [[String: String]]()
processImagesInDirectory(at: directoryPath, basePath: directoryPath, results: &results)
saveResultsToJSON(results, in: directoryPath)