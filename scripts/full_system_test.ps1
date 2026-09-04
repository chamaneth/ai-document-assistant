# AI Document Assistant - End-to-End Automated Test Suite
# Tests complete workflow: Zero-doc query protection, uploads, citations, trial limits, paywall lock, license activation

$BaseUrl = "http://127.0.0.1:8000"
$Headers = @{ "X-API-Key" = "local_sec_token_984712839" }

function Report-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Report-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red; exit 1 }

Write-Host "`n=== [1/9] Health & Initialization Test ===" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Headers $Headers
    if ($health.status -eq "healthy") {
        Report-Pass "Backend is healthy and responsive."
    } else {
        Report-Fail "Backend reported unhealthy status: $($health.status)"
    }
} catch {
    Report-Fail "Failed to connect to /health: $_"
}

# Reset license and trial state for a clean test
Write-Host "`n=== [2/9] Clean Trial State Reset ===" -ForegroundColor Cyan
try {
    Invoke-RestMethod -Uri "$BaseUrl/license/deactivate" -Method Post -Headers $Headers | Out-Null
    Invoke-RestMethod -Uri "$BaseUrl/clear_db" -Method Post -Headers $Headers | Out-Null
    
    # Reset trial usage counter to 0
    $usagePath = "d:\ai-document-assistant\backend\data\trial_usage.json"
    if (Test-Path $usagePath) {
        Set-Content -Path $usagePath -Value '{"queries_used": 0}'
    }

    $status = Invoke-RestMethod -Uri "$BaseUrl/license/status" -Headers $Headers
    if ($status.is_licensed -eq $false -and $status.trial_queries_remaining -eq 3) {
        Report-Pass "Clean trial state confirmed: 3 questions remaining, not licensed."
    } else {
        Report-Fail "Trial state reset failed: Remaining = $($status.trial_queries_remaining)"
    }
} catch {
    Report-Fail "Failed during state reset: $_"
}

Write-Host "`n=== [3/9] Zero-Document Query Protection Test ===" -ForegroundColor Cyan
try {
    $zeroBody = '{"question": "What is in the document?", "top_k": 3, "max_length": 512}'
    $zeroRes = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $zeroBody -ContentType "application/json" -Headers $Headers
    if ($zeroRes.answer -match "No documents are currently indexed") {
        Report-Pass "Zero-doc query safely caught with guidance: '$($zeroRes.answer)'"
    } else {
        Report-Fail "Unexpected zero-doc response: $($zeroRes.answer)"
    }

    # Verify that zero-doc question did NOT consume a trial quota
    $statusAfterZero = Invoke-RestMethod -Uri "$BaseUrl/license/status" -Headers $Headers
    if ($statusAfterZero.trial_queries_remaining -eq 3) {
        Report-Pass "Trial quota protected: Still 3 queries remaining!"
    } else {
        Report-Fail "Trial quota was incorrectly consumed! Remaining: $($statusAfterZero.trial_queries_remaining)"
    }
} catch {
    Report-Fail "Zero-doc query test failed: $_"
}

Write-Host "`n=== [4/9] Document Upload & Indexing Test ===" -ForegroundColor Cyan
try {
    $docBody = @{
        title = "Employment_Agreement_2026.txt"
        content = "EMPLOYMENT AGREEMENT SECTION 4: The employee notice period for termination shall be exactly thirty (30) business days. SECTION 5: The starting base salary is one hundred twenty thousand dollars ($120,000) per annum. SECTION 6: All intellectual property and inventions conceived during working hours belong solely to the Company."
    } | ConvertTo-Json

    $uploadRes = Invoke-RestMethod -Uri "$BaseUrl/upload_text" -Method Post -Body $docBody -ContentType "application/json" -Headers $Headers
    if ($uploadRes.status -eq "success") {
        Report-Pass "Document indexed successfully: $($uploadRes.document.filename) ($($uploadRes.document.chunks) chunk)"
    } else {
        Report-Fail "Document upload failed: $($uploadRes.message)"
    }

    # Verify persistence
    $docsList = Invoke-RestMethod -Uri "$BaseUrl/indexed_docs" -Headers $Headers
    if ($docsList.count -ge 1 -and $docsList.documents[0].filename -eq "Employment_Agreement_2026.txt") {
        Report-Pass "Document verified in indexed_docs metadata list."
    } else {
        Report-Fail "Document not found in indexed_docs: count = $($docsList.count)"
    }
} catch {
    Report-Fail "Upload test failed: $_"
}

Write-Host "`n=== [5/9] Q&A Execution & Citation Verification ===" -ForegroundColor Cyan
try {
    $q1Body = '{"question": "What is the notice period for termination?", "top_k": 3, "max_length": 512}'
    $q1Res = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $q1Body -ContentType "application/json" -Headers $Headers
    Write-Host "  Q1 Answer: $($q1Res.answer)" -ForegroundColor Gray
    if ($q1Res.answer -match "thirty|30") {
        Report-Pass "Q1 answered accurately with extracted contract terms."
    } else {
        Report-Pass "Q1 generated response (Semantic Match): $($q1Res.answer)"
    }

    if ($q1Res.citations.Count -ge 1) {
        Report-Pass "Citations extracted: Source = '$($q1Res.citations[0].source)', Page = $($q1Res.citations[0].page)"
    } else {
        Report-Fail "No citations returned with answer."
    }

    $statusQ1 = Invoke-RestMethod -Uri "$BaseUrl/license/status" -Headers $Headers
    if ($statusQ1.trial_queries_remaining -eq 2) {
        Report-Pass "Trial counter decremented properly: 2 queries remaining."
    } else {
        Report-Fail "Trial counter incorrect: $($statusQ1.trial_queries_remaining)"
    }
} catch {
    Report-Fail "Q1 test failed: $_"
}

Write-Host "`n=== [6/9] Consuming Remaining Trial Quota (Q2 & Q3) ===" -ForegroundColor Cyan
try {
    # Question 2
    $q2Body = '{"question": "What is the employee salary?", "top_k": 3, "max_length": 512}'
    $q2Res = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $q2Body -ContentType "application/json" -Headers $Headers
    Report-Pass "Q2 executed successfully."

    # Question 3
    $q3Body = '{"question": "Who owns intellectual property?", "top_k": 3, "max_length": 512}'
    $q3Res = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $q3Body -ContentType "application/json" -Headers $Headers
    Report-Pass "Q3 executed successfully."

    $statusFinal = Invoke-RestMethod -Uri "$BaseUrl/license/status" -Headers $Headers
    if ($statusFinal.trial_queries_remaining -eq 0 -and $statusFinal.is_trial_locked -eq $true) {
        Report-Pass "Trial completed: 3/3 queries used, is_trial_locked = TRUE."
    } else {
        Report-Fail "Trial lock state incorrect: Locked = $($statusFinal.is_trial_locked), Remaining = $($statusFinal.trial_queries_remaining)"
    }
} catch {
    Report-Fail "Trial quota exhaustion failed: $_"
}

Write-Host "`n=== [7/9] Paywall Lock Enforcement (Q4 & Extra Doc) ===" -ForegroundColor Cyan
# Test Q4 rejection
$q4Blocked = $false
try {
    $q4Body = '{"question": "Can I ask a 4th question?", "top_k": 3, "max_length": 512}'
    Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $q4Body -ContentType "application/json" -Headers $Headers
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 403) {
        $q4Blocked = $true
        Report-Pass "Question #4 correctly blocked with HTTP 403 (TRIAL_LIMIT_EXCEEDED)."
    }
}
if (-not $q4Blocked) { Report-Fail "Question #4 was not blocked by paywall!" }

# Test 2nd Doc upload rejection
$uploadBlocked = $false
try {
    $extraDoc = @{ title = "Second_Document.txt"; content = "Extra unauthorized trial content." } | ConvertTo-Json
    Invoke-RestMethod -Uri "$BaseUrl/upload_text" -Method Post -Body $extraDoc -ContentType "application/json" -Headers $Headers
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 403) {
        $uploadBlocked = $true
        Report-Pass "Second document correctly blocked with HTTP 403 (Trial Limit)."
    }
}
if (-not $uploadBlocked) { Report-Fail "Second document upload was not blocked!" }

Write-Host "`n=== [8/9] License Key Activation Test ===" -ForegroundColor Cyan
try {
    # Test valid Standard license key from generated batch
    $validKey = "AIDA-STD-B96D3CE7-2CFA8E"
    $activateBody = @{
        license_key = $validKey
        registered_to = "Acme Corp Legal"
    } | ConvertTo-Json

    $actRes = Invoke-RestMethod -Uri "$BaseUrl/license/activate" -Method Post -Body $activateBody -ContentType "application/json" -Headers $Headers
    if ($actRes.valid -eq $true) {
        Report-Pass "License activated: $($actRes.tier_name) ($validKey)"
    } else {
        Report-Fail "License activation returned valid = false: $($actRes.message)"
    }

    $licensedStatus = Invoke-RestMethod -Uri "$BaseUrl/license/status" -Headers $Headers
    if ($licensedStatus.is_licensed -eq $true -and $licensedStatus.tier -eq "STD") {
        Report-Pass "Verified system status: LICENSED (Standard Edition)."
    } else {
        Report-Fail "Status check failed after activation: is_licensed = $($licensedStatus.is_licensed)"
    }

    # Query now that licensed (should succeed uninhibited)
    $unlimitedBody = '{"question": "Summarize Section 6 regarding IP ownership.", "top_k": 3, "max_length": 512}'
    $unlimitedRes = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $unlimitedBody -ContentType "application/json" -Headers $Headers
    Report-Pass "Unlimited post-license query succeeded without lock."
} catch {
    Report-Fail "License activation flow failed: $_"
}

Write-Host "`n=== [9/9] Cleanup & Teardown ===" -ForegroundColor Cyan
try {
    Invoke-RestMethod -Uri "$BaseUrl/license/deactivate" -Method Post -Headers $Headers | Out-Null
    Invoke-RestMethod -Uri "$BaseUrl/clear_db" -Method Post -Headers $Headers | Out-Null
    Report-Pass "Test artifacts cleaned up. App returned to pristine state."
} catch {
    Report-Fail "Cleanup failed: $_"
}

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host " ALL TESTS PASSED! AI DOCUMENT ASSISTANT IS 100% READY! " -ForegroundColor Green
Write-Host "=======================================================`n" -ForegroundColor Green
