// MV3 service worker. Opens the side panel on toolbar click or the Alt+Shift+S hotkey.

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error)

chrome.commands.onCommand.addListener((command, tab) => {
  if (command !== 'lookup-sign' || tab?.windowId === undefined) return
  // sidePanel.open must run inside the hotkey's user gesture, so call it before anything async.
  chrome.sidePanel.open({ windowId: tab.windowId }).catch(console.error)
  // TODO: capture the last few seconds via tabCapture + offscreen document (add those permissions then).
})
