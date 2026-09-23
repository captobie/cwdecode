import Testing
@testable import CWDecode

struct AudioDevicesTests {
    /// Runs against whatever hardware is attached, so it only checks invariants.
    @Test func listsInputDevicesConsistently() {
        let devices = AudioDevices.inputDevices()
        print("Input devices:", devices.map(\.name), "default:", AudioDevices.defaultInputDevice()?.name ?? "none")
        #expect(Set(devices.map(\.uid)).count == devices.count)
        #expect(devices.allSatisfy { !$0.name.isEmpty && !$0.uid.isEmpty })
        if let defaultDevice = AudioDevices.defaultInputDevice() {
            #expect(devices.contains(defaultDevice))
        }
    }
}
