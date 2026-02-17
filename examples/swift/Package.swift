// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "InvisibleCudaProof",
    targets: [
        .executableTarget(
            name: "InvisibleCudaProof",
            path: "Sources/InvisibleCudaProof",
            linkerSettings: [.linkedLibrary("dl")]
        ),
    ]
)
