"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Sparkles, Send, BarChart3, TrendingUp, Database, Zap, Upload, Link2, X, ImageIcon, File } from "lucide-react"

type UploadedFile = {
  id: string
  file: File
  name: string
  type: string
  size: number
}

type ContextLink = {
  id: string
  url: string
}

export default function AlvynAgent() {
  const [query, setQuery] = useState("")
  const [instructions, setInstructions] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [contextLinks, setContextLinks] = useState<ContextLink[]>([])
  const [linkInput, setLinkInput] = useState("")
  const [showLinkInput, setShowLinkInput] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string>("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      setError("Please enter analysis questions.")
      return
    }

    setIsProcessing(true)
    setError("")
    setResults(null)

    try {
      const formData = new FormData()

      let questionsContent = query
      if (instructions.trim()) {
        questionsContent += `\n\nAdditional Instructions:\n${instructions}`
      }

      if (contextLinks.length > 0) {
        questionsContent += `\n\nContext Links:\n${contextLinks.map((link) => `• ${link.url}`).join("\n")}`
      }

      const questionsBlob = new Blob([questionsContent], { type: "text/plain" })
      formData.append("questions.txt", questionsBlob, "questions.txt")

      uploadedFiles.forEach((fileObj) => {
        formData.append(fileObj.file.name, fileObj.file, fileObj.file.name)
      })

      console.log("[Alvyn] Sending form data with files:", Array.from(formData.keys()))

      // POST to our API route which proxies to FastAPI backend
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      console.log("[Alvyn] Received result:", result)

      if (result.error) {
        setError(result.error)
      } else {
        setResults(result)
      }
    } catch (err: any) {
      console.error("[Alvyn] Error:", err)
      setError(err.message || "An error occurred while processing your request.")
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    const newFiles: UploadedFile[] = Array.from(files).map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      file: file,
      name: file.name,
      type: file.type,
      size: file.size,
    }))

    setUploadedFiles((prev) => [...prev, ...newFiles])
  }

  const handleAddLink = () => {
    if (!linkInput.trim()) return

    const newLink: ContextLink = {
      id: Math.random().toString(36).substr(2, 9),
      url: linkInput,
    }

    setContextLinks((prev) => [...prev, newLink])
    setLinkInput("")
    setShowLinkInput(false)
  }

  const removeFile = (id: string) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== id))
  }

  const removeLink = (id: string) => {
    setContextLinks((prev) => prev.filter((link) => link.id !== id))
  }

  const getFileIcon = (type: string) => {
    if (type.includes("image")) return <ImageIcon className="h-4 w-4" />
    if (type.includes("csv") || type.includes("excel") || type.includes("spreadsheet"))
      return <BarChart3 className="h-4 w-4" />
    if (type.includes("json")) return <Database className="h-4 w-4" />
    return <File className="h-4 w-4" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const capabilities = [
    { icon: BarChart3, label: "Data Analysis", description: "Advanced statistical analysis and insights" },
    { icon: TrendingUp, label: "Trend Detection", description: "Identify patterns and predict outcomes" },
    { icon: Database, label: "Query Optimization", description: "Intelligent data retrieval and processing" },
    { icon: Zap, label: "Real-time Processing", description: "Lightning-fast computation and results" },
  ]

  return (
    <div className="min-h-screen bg-background dark">
      <div className="fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-background to-violet-900/20" />
        <div className="absolute top-0 -left-40 h-80 w-80 rounded-full bg-purple-600/20 blur-3xl animate-pulse" />
        <div className="absolute top-1/4 -right-40 h-96 w-96 rounded-full bg-violet-600/20 blur-3xl animate-pulse delay-1000" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-fuchsia-600/20 blur-3xl animate-pulse delay-500" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(139,92,246,0.1)_1px,transparent_1px),linear-gradient(to_bottom,rgba(139,92,246,0.1)_1px,transparent_1px)] bg-[size:4rem_4rem]" />
      </div>

      <div className="relative">
        <header className="border-b border-white/10 backdrop-blur-xl bg-background/30">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500/20 to-violet-600/20 backdrop-blur-sm ring-1 ring-white/20">
                  <Sparkles className="h-5 w-5 text-purple-300" />
                  <div className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-violet-400 animate-pulse shadow-lg shadow-violet-400/50" />
                </div>
                <div>
                  <h1 className="text-xl font-semibold bg-gradient-to-r from-purple-200 via-violet-200 to-fuchsia-200 bg-clip-text text-transparent">
                    Alvyn
                  </h1>
                  <p className="text-xs text-purple-300/60">AI Data Analyst</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" className="text-purple-200/60 hover:text-purple-100 hover:bg-white/5">
                  Docs
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-white/10 bg-white/5 backdrop-blur-sm hover:bg-white/10 text-purple-100"
                >
                  Sign In
                </Button>
              </div>
            </div>
          </div>
        </header>

        <section className="container mx-auto px-4 py-20 md:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-purple-400/20 bg-purple-500/10 backdrop-blur-sm px-4 py-1.5 text-sm text-purple-200">
              <Zap className="h-3.5 w-3.5" />
              <span className="font-medium">Lightning Fast Analysis</span>
            </div>

            <h2 className="mb-6 text-5xl font-bold tracking-tight text-foreground md:text-7xl text-balance">
              Your AI-Powered
              <span className="bg-gradient-to-r from-purple-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent animate-gradient">
                {" "}
                Data Analyst
              </span>
            </h2>

            <p className="mb-12 text-lg leading-relaxed text-purple-200/70 md:text-xl text-balance">
              Transform complex datasets into actionable insights with natural language. Alvyn processes millions of
              data points in seconds, delivering precise analytics you can trust.
            </p>

            <div className="mx-auto max-w-2xl space-y-4">
              {showLinkInput && (
                <div className="relative group">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-500 to-fuchsia-600 rounded-xl opacity-30 blur transition duration-300" />
                  <div className="relative flex gap-2">
                    <Input
                      type="url"
                      placeholder="Paste link for context..."
                      value={linkInput}
                      onChange={(e) => setLinkInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault()
                          handleAddLink()
                        }
                      }}
                      className="h-12 text-base bg-black/40 backdrop-blur-xl border-white/10 text-purple-50 placeholder:text-purple-200/40 focus:border-violet-400/50 focus:ring-2 focus:ring-violet-400/20 rounded-xl"
                    />
                    <Button
                      onClick={handleAddLink}
                      size="icon"
                      className="h-12 w-12 rounded-xl bg-gradient-to-r from-violet-500 to-fuchsia-600 hover:from-violet-600 hover:to-fuchsia-700 shadow-lg shadow-violet-500/30"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                    <Button
                      onClick={() => {
                        setShowLinkInput(false)
                        setLinkInput("")
                      }}
                      size="icon"
                      variant="ghost"
                      className="h-12 w-12 rounded-xl text-purple-300 hover:text-purple-100 hover:bg-white/5"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}

              <form onSubmit={handleSubmit}>
                <div className="relative group mb-4">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-violet-600 rounded-2xl opacity-30 group-hover:opacity-50 blur transition duration-300" />
                  <div className="relative">
                    <textarea
                      placeholder="Enter your analysis questions here..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      rows={6}
                      className="w-full min-h-[160px] max-h-[300px] resize-y pl-4 pr-4 pt-4 pb-14 text-base bg-black/40 backdrop-blur-xl border border-white/10 text-purple-50 placeholder:text-purple-200/40 focus:border-purple-400/50 focus:ring-2 focus:ring-purple-400/20 rounded-2xl focus-visible:outline-none"
                    />
                    <div className="absolute right-2 bottom-2 flex gap-1">
                      <label className="cursor-pointer">
                        <input
                          type="file"
                          multiple
                          accept=".csv,.json,.xlsx,.xls,.png,.jpg,.jpeg,.gif,.webp,.txt,.tsv,.parquet"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                        <div className="h-10 w-10 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 hover:border-purple-400/30 transition-all flex items-center justify-center group/upload">
                          <Upload className="h-4 w-4 text-purple-300 group-hover/upload:text-purple-100" />
                        </div>
                      </label>

                      <Button
                        type="button"
                        size="icon"
                        onClick={() => setShowLinkInput(!showLinkInput)}
                        className="h-10 w-10 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 hover:border-violet-400/30"
                      >
                        <Link2 className="h-4 w-4 text-violet-300" />
                      </Button>

                      <Button
                        type="submit"
                        size="icon"
                        disabled={isProcessing}
                        className="h-10 w-10 rounded-xl bg-gradient-to-r from-purple-500 to-violet-600 hover:from-purple-600 hover:to-violet-700 shadow-lg shadow-purple-500/30"
                      >
                        {isProcessing ? (
                          <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        ) : (
                          <Send className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </div>

                <p className="mt-3 text-xs text-purple-300/50">
                  Upload CSV, JSON, XLSX, or images • Add links for context • Ask any data question
                </p>
              </form>

              {uploadedFiles.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {uploadedFiles.map((file) => (
                    <div
                      key={file.id}
                      className="group flex items-center gap-2 rounded-lg border border-purple-400/20 bg-purple-500/10 backdrop-blur-sm px-3 py-2 text-sm"
                    >
                      <div className="text-purple-300">{getFileIcon(file.type)}</div>
                      <span className="text-purple-100">{file.name}</span>
                      <span className="text-purple-300/60 text-xs">({formatFileSize(file.size)})</span>
                      <button
                        onClick={() => removeFile(file.id)}
                        className="ml-1 text-purple-300/60 hover:text-purple-100 transition-colors"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {contextLinks.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {contextLinks.map((link) => (
                    <div
                      key={link.id}
                      className="group flex items-center gap-2 rounded-lg border border-violet-400/20 bg-violet-500/10 backdrop-blur-sm px-3 py-2 text-sm"
                    >
                      <Link2 className="h-4 w-4 text-violet-300" />
                      <span className="text-violet-100 truncate max-w-xs">{link.url}</span>
                      <button
                        onClick={() => removeLink(link.id)}
                        className="ml-1 text-violet-300/60 hover:text-violet-100 transition-colors"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {isProcessing && (
                <div className="mt-8 text-center">
                  <div className="inline-flex items-center gap-3 rounded-xl border border-purple-400/20 bg-purple-500/10 backdrop-blur-sm px-6 py-4">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-purple-300 border-t-transparent" />
                    <span className="text-purple-200">
                      Generating and executing analysis code... This may take up to 3 minutes
                    </span>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-8 rounded-xl border border-red-400/20 bg-red-500/10 backdrop-blur-sm px-6 py-4">
                  <p className="text-red-200">{error}</p>
                </div>
              )}

              {results && (
                <div className="mt-8 space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="rounded-xl border border-purple-400/20 bg-purple-500/10 backdrop-blur-sm px-6 py-4 text-center">
                    <p className="text-purple-200 font-medium">Analysis Complete</p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/40 backdrop-blur-xl p-6 max-h-[800px] overflow-y-auto space-y-6">
                    {Object.entries(results).map(([key, value]) => {
                      // Check if value is a base64 image
                      const isBase64Image = typeof value === "string" && (
                        value.startsWith("data:image/") ||
                        value.startsWith("iVBOR") ||
                        value.startsWith("/9j/")
                      )
                      
                      // Skip error keys that are empty
                      if (key.endsWith("_error") && !value) return null

                      // Helper to render object as nice key-value pairs
                      const renderValue = (val: any): React.ReactNode => {
                        if (val === null || val === undefined) return <span className="text-purple-300/50">—</span>
                        if (typeof val === "boolean") return <span className={val ? "text-green-400" : "text-red-400"}>{val ? "Yes" : "No"}</span>
                        if (typeof val === "number") return <span className="text-violet-300 font-medium">{val.toLocaleString()}</span>
                        if (typeof val === "string") return <span className="text-purple-100">{val}</span>
                        if (Array.isArray(val)) {
                          if (val.length === 0) return <span className="text-purple-300/50">Empty list</span>
                          if (typeof val[0] === "object") {
                            return (
                              <div className="space-y-2 mt-1">
                                {val.map((item, i) => (
                                  <div key={i} className="bg-black/20 rounded-lg p-2 text-sm">
                                    {renderValue(item)}
                                  </div>
                                ))}
                              </div>
                            )
                          }
                          return <span className="text-purple-100">{val.join(", ")}</span>
                        }
                        if (typeof val === "object") {
                          return (
                            <div className="grid gap-2 mt-1">
                              {Object.entries(val).map(([k, v]) => (
                                <div key={k} className="flex items-start gap-3 bg-black/20 rounded-lg px-3 py-2">
                                  <span className="text-purple-400 text-sm font-medium min-w-[120px] capitalize">
                                    {k.replace(/_/g, " ")}:
                                  </span>
                                  <span className="text-purple-100 text-sm">{renderValue(v)}</span>
                                </div>
                              ))}
                            </div>
                          )
                        }
                        return <span className="text-purple-100">{String(val)}</span>
                      }
                      
                      return (
                        <div key={key} className="border-b border-white/10 pb-4 last:border-0 last:pb-0">
                          <h4 className="text-purple-300 font-semibold text-sm uppercase tracking-wide mb-2">
                            {key.replace(/_/g, " ")}
                          </h4>
                          {isBase64Image ? (
                            <div className="mt-2">
                              <img
                                src={typeof value === "string" && value.startsWith("data:") ? value : `data:image/png;base64,${value}`}
                                alt={key}
                                className="max-w-full h-auto rounded-lg border border-white/10"
                              />
                            </div>
                          ) : (
                            <div className="text-purple-100 text-sm">
                              {renderValue(value)}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 pb-20 md:pb-32">
          <div className="mb-12 text-center">
            <h3 className="mb-3 text-sm font-medium uppercase tracking-wider text-purple-300">Capabilities</h3>
            <p className="text-2xl font-semibold text-foreground md:text-3xl">
              Everything you need for data excellence
            </p>
          </div>

          <div className="mx-auto grid max-w-5xl gap-4 md:grid-cols-2 lg:gap-6">
            {capabilities.map((capability, index) => {
              const Icon = capability.icon
              return (
                <Card
                  key={index}
                  className="group relative overflow-hidden border-white/10 bg-white/5 backdrop-blur-xl transition-all hover:border-purple-400/30 hover:shadow-xl hover:shadow-purple-500/10"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-violet-600/10 opacity-0 transition-opacity group-hover:opacity-100" />
                  <div className="relative p-6">
                    <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500/20 to-violet-600/20 backdrop-blur-sm ring-1 ring-white/20 transition-transform group-hover:scale-110">
                      <Icon className="h-6 w-6 text-purple-300" />
                    </div>
                    <h4 className="mb-2 text-lg font-semibold text-foreground">{capability.label}</h4>
                    <p className="text-sm leading-relaxed text-purple-200/60">{capability.description}</p>
                  </div>
                </Card>
              )
            })}
          </div>
        </section>

        <section className="border-t border-white/10 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-16">
            <div className="grid gap-8 md:grid-cols-3 lg:gap-12">
              {[
                { value: "10M+", label: "Data Points Analyzed" },
                { value: "<100ms", label: "Average Query Time" },
                { value: "99.9%", label: "Accuracy Rate" },
              ].map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="mb-2 text-4xl font-bold bg-gradient-to-r from-purple-400 to-violet-400 bg-clip-text text-transparent md:text-5xl">
                    {stat.value}
                  </div>
                  <div className="text-sm text-purple-200/60 md:text-base">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>

      <style jsx global>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s linear infinite;
        }
      `}</style>
    </div>
  )
}
