import { NextRequest, NextResponse } from "next/server"

// Proxy requests to the FastAPI backend
// Use NEXT_PUBLIC_BACKEND_URL at runtime, fallback to localhost during development
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData()

    // Forward the request to FastAPI backend
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[API Route] Backend error:", response.status, errorText)
      return NextResponse.json(
        { error: `Backend error: ${response.status}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error("[API Route] Error:", error)
    return NextResponse.json(
      { error: error.message || "Failed to connect to backend" },
      { status: 500 }
    )
  }
}
