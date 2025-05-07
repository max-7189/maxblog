# PhotoManager: Organizing Memories, Simplifying Life


# PhotoManager: Organizing Memories, Simplifying Life

## Project Origins: A Journey to Find Memories

It all began on an ordinary evening when a casual conversation with an old friend led me on a journey to find memories. My friend mentioned a photo we had taken together years ago, and in that moment, those laughs and scenes felt as if they had happened just yesterday. I was eager to find that photo and relive those wonderful moments.

However, reality quickly gave me a harsh reminder—my photo library was cluttered with thousands of pictures, without any organization or structure. I had to resort to the most primitive method: scrolling through them one by one.

That night, lying in bed, I spent two full hours browsing through photos. My finger constantly swiped the screen as my eyes scanned image after image. During this process, I experienced an emotional rollercoaster:

 Seeing old friends' smiling faces made me smile involuntarily;
 
 Glimpsing friends I had lost contact with for years brought a tinge of melancholy;

 Viewing places I had once visited made those landscapes reappear before my eyes.

But at the same time, I discovered that my photo library was filled with numerous useless images: work screenshots, blurry photos, duplicate images... Not only did they occupy my storage space, but they also hindered me from finding truly precious memories.

It was at that moment that an idea flashed through my mind: Why not organize my photo library while browsing? Keep only the memories truly worth preserving and delete those insignificant images. This way, the next time I wanted to revisit old times, the journey would be smoother and more pure.

With this idea in mind, the "PhotoManager" project was born.

## Technical Vision: Not Just a Photo Browser

From the beginning, I hoped to create not just an ordinary photo application, but a tool that could truly help people manage their memories. I set several core objectives for this project:

1. **Smooth Browsing Experience**: The flow of memories should be seamless, without lag or stuttering
2. **Intelligent Classification System**: Organize photos along a timeline, making memories traceable
3. **Efficient Marking Function**: Easily mark photos to keep or delete
4. **Optimized Memory Management**: Maintain stability even when dealing with thousands of photos

## Technical Depth: Building a Reliable Photo Management System

### Core Architecture: Flexible Application of the MVVM Pattern

The project adopts the MVVM (Model-View-ViewModel) architectural pattern, which not only creates a clear code structure but also achieves good separation between the UI layer and business logic layer. This separation becomes particularly important in applications handling large volumes of photos.

```swift
class PhotoViewModel: ObservableObject {
    @Published var currentIndex = 0
    @Published var showingDeleteConfirmation = false
    
    private let photoManager: PhotoManager
    
    // Expose necessary data to the view
    var mediaItems: [MediaItem] { photoManager.mediaItems }
    var isInitialLoading: Bool { photoManager.isInitialLoading }
    var loadingProgress: Float { photoManager.loadingProgress }
    var pendingDeletionsCount: Int { mediaItems.filter { $0.markStatus == .delete }.count }
    
    // Other methods...
}
```

### Photo Loading System: Balancing Performance and Experience

During development, the biggest technical challenge was how to handle a large number of high-resolution photos. If all original images were loaded directly, the application would quickly exhaust memory; if only thumbnails were loaded, the user experience would suffer.

To address this, I designed a multi-tiered image loading system:

1. **Thumbnails**: All photos are first loaded as thumbnails, ensuring quick display
2. **Preview Images**: When a user swipes to a photo, a medium-quality preview image is loaded
3. **Full-Quality Images**: When a user pauses at a particular photo, a full-quality image is loaded

```swift
enum ImageQuality {
    case thumbnail     // For thumbnail strips, fixed small size
    case preview       // Preview image, medium quality, fast loading
    case fullQuality   // Full quality, loaded when user lingers
}
```

### Memory Management: Window Loading Mechanism

To solve memory issues, I implemented a "window loading" mechanism. Simply put, the application only keeps a certain number of photos before and after the user's current viewing position in memory, while other photos are released.

```swift
private func updateWindow(currentIndex: Int) {
    // Calculate range to keep: 50 photos before and after current position
    let start = max(0, currentIndex - PhotoLoadingConfig.highQualityWindow)
    let end = min(mediaItems.count, currentIndex + PhotoLoadingConfig.highQualityWindow + 1)
    let keepRange = start..<end
    
    // Release memory for items outside range
    releaseMemoryForItemsOutside(keepRange)
    
    // Update current window start position
    currentWindowStart = keepRange.lowerBound
}
```

This mechanism ensures that even when facing a library with tens of thousands of photos, the application can maintain smooth operation without crashing or lagging.

### User Experience: Details Determine Success

When designing the user interface, I paid special attention to those small but crucial details:

1. **Progressive Loading**: The transition of photos from blurry to sharp is smooth, almost imperceptible to users
2. **Gesture Operations**: Mark photos through intuitive swipe gestures—swipe up to delete, down to keep
3. **Visual Feedback**: Marked photos have clear visual indicators, allowing users to immediately know which photos will be deleted
4. **Batch Operations**: Support for deleting multiple marked photos at once, improving efficiency

```swift
private var overlayButtons: some View {
    VStack {
        Spacer()
        HStack {
            Button(action: { viewModel.markForDeletion(at: viewModel.currentIndex) }) {
                Image(systemName: "trash")
                    .font(.title)
                    .foregroundColor(.red)
                    .padding()
                    .background(Circle().fill(.ultraThinMaterial))
            }
            
            Spacer()
            
            Button(action: { viewModel.keepCurrentPhoto(at: viewModel.currentIndex) }) {
                Image(systemName: "star")
                    .font(.title)
                    .foregroundColor(.yellow)
                    .padding()
                    .background(Circle().fill(.ultraThinMaterial))
            }
        }
        .padding(.horizontal, 40)
        .padding(.bottom, 20)
    }
}
```

### Data Persistence: Never Lose Any Marks

Core Data was used to implement the persistence of marking states, ensuring that users' marks would not be lost when the application restarts:

```swift
private func saveMarkForItem(_ item: MediaItem) {
    let context = CoreDataManager.shared.context
    context.performAndWait {
        let fetchRequest: NSFetchRequest<MediaItemEntity> = MediaItemEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %@", item.id)
        
        do {
            let results = try context.fetch(fetchRequest)
            let entity = results.first ?? MediaItemEntity(context: context)
            entity.id = item.id
            entity.markStatus = Int16(item.markStatus.rawValue)
            
            if context.hasChanges {
                try context.save()
            }
        } catch {
            print("Failed to save mark: \(error)")
        }
    }
}
```

## Development Process: Challenges and Growth

The process of developing this application was not smooth sailing. The initial version was simple and straightforward but frequently crashed when handling a large number of photos. Through repeated testing and optimization, I gradually improved the memory management mechanism and loading strategy.

Particularly when implementing video preview functionality, I encountered numerous challenges. iOS's video loading mechanism differs significantly from that of images and requires special handling:

```swift
func playVideo() {
    print("[VideoCardViewModel] Attempting to play video - ID: \(mediaItem.id)")
    if let player = player, isPlayerReady {
        print("[VideoCardViewModel] Starting playback - ID: \(mediaItem.id)")
        player.play()
    } else {
        print("[VideoCardViewModel] Waiting for player to be ready - ID: \(mediaItem.id)")
    }
}
```

Each problem solved gave me a deeper understanding of iOS's image handling system.

## Future Vision: AI-Powered Photo Management with FastViT

### The New Era of Intelligent Photo Analysis

As the project evolved, I increasingly recognized the enormous potential of artificial intelligence in the field of photo management. After multiple conversations with users, I discovered a common need: they all wanted photo management to no longer rely on manual organization but instead automatically identify photo content and categorize it. This is precisely the transformation that FastViT technology will bring us.

FastViT, as a lightweight and efficient Vision Transformer model, not only performs excellently in image recognition accuracy compared to traditional CNN models, but more importantly, its computational efficiency is high enough to allow local inference on mobile devices. This means users' photos don't need to be uploaded to the cloud for intelligent analysis.

### FastViT Integration Plan

#### Phase One: Basic Recognition Capabilities

In the first phase of integrating FastViT, I plan to implement the following core functions:

```swift
class AIPhotoAnalyzer {
    private let fastVitModel: FastVitModel
    
    func analyzePhoto(_ photo: UIImage) async throws -> PhotoAnalysisResult {
        // Analyze photo using FastViT model
        let features = try await fastVitModel.extractFeatures(from: photo)
        
        // Classify based on features
        let categories = classifyImage(features)
        let objects = detectObjects(features)
        let sceneType = determineSceneType(features)
        
        return PhotoAnalysisResult(
            categories: categories,
            detectedObjects: objects,
            sceneType: sceneType,
            quality: assessImageQuality(photo)
        )
    }
    
    // Other auxiliary methods...
}
```

This will add intelligent tags to each photo, including:

1. **Scene Recognition**: Automatically identify the type of scene captured in the photo (beach, city, forest, indoor, etc.)
2. **Object Detection**: Identify the main objects in the photo (people, animals, buildings, food, etc.)
3. **Photo Quality Assessment**: Automatically evaluate the clarity, exposure, and composition quality of the photo

#### Phase Two: Advanced Facial and Group Analysis

```swift
extension AIPhotoAnalyzer {
    func extractFacialFeatures(_ photo: UIImage) async throws -> [FaceFeature] {
        // Use FastViT for facial feature extraction
        let faces = try await fastVitModel.detectFaces(in: photo)
        return faces.map { face in
            return FaceFeature(
                boundingBox: face.boundingBox,
                landmarks: face.landmarks,
                embedding: face.embedding,
                expressions: analyzeExpressions(face)
            )
        }
    }
    
    func groupSimilarFaces(_ faces: [FaceFeature]) -> [FaceGroup] {
        // Use clustering algorithms to group similar faces
        // Return face grouping results
    }
}
```

This phase will introduce more complex analytical capabilities:

1. **Facial Recognition and Grouping**: Identify faces in photos and automatically group faces of the same person in different photos, making it easy for users to find photos of specific individuals
2. **Expression and Emotion Analysis**: Recognize expressions and emotions of people in photos, helping users find photos from joyful or special moments
3. **Social Relationship Graph**: Build a social relationship graph based on the frequency of people appearing together in photos, helping users better organize social memories

#### Phase Three: Intelligent Organization and Narrative

```swift
class MemoryStoryGenerator {
    private let aiAnalyzer: AIPhotoAnalyzer
    private let narrativeModel: NarrativeModel
    
    func generateStory(from photos: [AnalyzedPhoto], theme: StoryTheme? = nil) async throws -> MemoryStory {
        // Select key photos
        let keyPhotos = selectKeyPhotos(photos)
        
        // Determine storyline
        let storyLine = determineStoryline(keyPhotos, theme: theme)
        
        // Generate story title and description
        let storyTitle = try await narrativeModel.generateTitle(for: storyLine)
        let storyDescription = try await narrativeModel.generateDescription(for: storyLine)
        
        return MemoryStory(
            title: storyTitle,
            description: storyDescription,
            photos: arrangePhotosForStory(keyPhotos, storyline: storyLine),
            coverPhoto: selectCoverPhoto(keyPhotos)
        )
    }
    
    // Helper methods...
}
```

The final phase will introduce the most innovative features:

1. **Intelligent Story Generation**: The system will analyze photos that are close in time, same in location, or related in theme to automatically generate "memory stories," such as "Summer Beach Trip 2023"
2. **Key Moment Capture**: Identify and highlight photos of important moments from a large collection, such as the candle-blowing moment at a birthday party
3. **Photo Quality Sorting**: Automatically select photos with the best composition and highest clarity among similar photos, helping users quickly filter
4. **Personalized Recommendation System**: Learn users' photo preferences based on their browsing and saving habits, providing personalized photo recommendations

### Technical Challenges and Solutions

Integrating FastViT is no easy task, and we will face several major challenges:

#### 1. Mobile Performance Optimization

Although FastViT is relatively lightweight, running complex neural networks on mobile phones remains challenging. I plan to adopt the following strategies:

```swift
class ModelOptimizer {
    func optimizeModel() -> OptimizedModel {
        // Model quantization, converting floating-point weights to 8-bit or 16-bit integers
        let quantizedModel = quantizeModel(originalModel)
        
        // Model pruning, removing unnecessary connections
        let prunedModel = pruneModel(quantizedModel)
        
        // Compile optimization using CoreML
        return compileCoreMLModel(prunedModel)
    }
}
```

- **Model Quantization and Pruning**: Reduce model size and memory usage
- **Asynchronous Processing**: Execute analysis tasks in background threads to ensure UI fluidity
- **Batch Processing Mechanism**: Analyze multiple photos at once to improve throughput
- **Tiered Analysis**: Perform simple, quick analysis first, and conduct deeper analysis when users view specific photos

#### 2. Privacy Protection

User photos often contain private information, making privacy protection crucial:

```swift
class PrivacyManager {
    func processImageWithPrivacyProtection(_ image: UIImage) -> UIImage {
        // Blur sensitive areas such as IDs, credit cards, etc.
        let sensitiveRegions = detectSensitiveRegions(image)
        
        // Store analysis results locally only, don't upload original photos
        return blurRegions(image, regions: sensitiveRegions)
    }
}
```

- **Local Processing**: All photo analysis completed locally on the device, not uploaded to the cloud
- **Sensitive Information Detection**: Automatically identify and blur sensitive information in photos, such as documents and IDs
- **User Control**: Allow users to turn off specific types of analysis features at any time
- **Data Isolation**: Analysis results stored only on the user's device, not shared with other services

#### 3. Continuous Learning and Personalization

Each user's photo habits and preferences differ, and the system needs to adapt to these differences:

```swift
class PersonalizedLearningSystem {
    private var userPreferenceModel: UserPreferenceModel
    
    func updateModelBasedOnUserFeedback(_ feedback: UserFeedback) {
        // Update preference model based on user's actions with photos
        userPreferenceModel.update(with: feedback)
        
        // Adjust classification thresholds and weights
        adjustClassificationParameters(based: userPreferenceModel)
    }
}
```

- **Feedback Learning**: Adjust algorithms based on user behavior of keeping/deleting photos
- **Preference Modeling**: Build user photo preference models for personalized sorting and recommendations
- **Incremental Learning**: Models can continuously optimize with use without requiring complete retraining
- **Multi-scenario Adaptation**: Adjust analysis strategies for different user scenarios (family photos, travel photos, work documents, etc.)



### Future AI-Driven User Experience

Imagine what the user experience will be like when FastViT is fully integrated:

John opens PhotoManager, and the system automatically presents several featured "memory stories" for him—"Last Weekend's Family Barbecue," "Last Summer's European Trip," "Daughter's Growth Record." These are all automatically generated by the system based on the time, location, and content of the photos.

When he wants to organize his albums, he no longer needs to view them one by one. The system has intelligently marked photo quality—clear, well-composed photos are marked as "keep," while blurry, duplicate photos are suggested for "deletion."

He searches for "photos with Sarah," and the system immediately displays all photos containing him and Sarah, even if these photos were never manually tagged.

The entire experience will transform from passive manual organization to active intelligent recommendations, returning photo management to its original purpose—appreciating and sharing beautiful memories, rather than being troubled by tedious organization work.

## Conclusion: A Project About Memories

"PhotoManager" is not just a technical project; it's a project about memories. Each photo carries a story, a moment, an emotion. Through this application, I hope people can better preserve and organize these precious fragments of memory.

That two-hour journey searching for memories that night eventually became the starting point for this project. From a simple idea to a complex implementation, from personal needs to a universal solution, the process was full of challenges but also full of gains.

I believe that when you open this application and start organizing your photos, you'll also embark on a beautiful journey of memories. Those forgotten smiles, those distant landscapes, those precious moments will all reappear before your eyes, allowing you to revisit those fragments of life worth treasuring.

That is the meaning of "PhotoManager": not just managing photos, but managing memories, managing emotions, managing those beautiful moments in life. With the integration of AI capabilities, this vision will be further realized, making every memory easily found and giving every photo its rightful place. 
