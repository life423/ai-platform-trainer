#!/bin/bash
# Deployment Monitoring Script for AI Platform Trainer
# Uses GitHub CLI to monitor and manage deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${2}${1}${NC}"
}

# Function to get latest workflow run
get_latest_run() {
    gh run list --workflow="Build and Deploy Executables" --limit 1 --json databaseId,status,conclusion,url | jq -r '.[] | "\(.databaseId) \(.status) \(.conclusion) \(.url)"'
}

# Function to monitor workflow progress
monitor_workflow() {
    print_status "🔍 Monitoring latest deployment workflow..." "$BLUE"
    
    while true; do
        run_info=$(get_latest_run)
        run_id=$(echo $run_info | cut -d' ' -f1)
        status=$(echo $run_info | cut -d' ' -f2)
        conclusion=$(echo $run_info | cut -d' ' -f3)
        url=$(echo $run_info | cut -d' ' -f4)
        
        clear
        print_status "🚀 AI Platform Trainer - Deployment Monitor" "$BLUE"
        print_status "=============================================" "$BLUE"
        echo
        print_status "📋 Run ID: $run_id" "$NC"
        print_status "🔗 URL: $url" "$NC"
        echo
        
        case $status in
            "queued")
                print_status "⏳ Status: Queued" "$YELLOW"
                ;;
            "in_progress")
                print_status "🔄 Status: In Progress" "$BLUE"
                echo
                print_status "Job Status:" "$NC"
                gh run view $run_id --json jobs | jq -r '.jobs[] | "  \(.name): \(.status) \(.conclusion // "running")"'
                ;;
            "completed")
                if [ "$conclusion" = "success" ]; then
                    print_status "✅ Status: Completed Successfully!" "$GREEN"
                    echo
                    print_status "🎉 Deployment successful! Checking artifacts..." "$GREEN"
                    gh run view $run_id --json artifacts | jq -r '.artifacts[] | "  📦 \(.name) (\(.size_in_bytes) bytes)"'
                    
                    # Check if release was created
                    latest_release=$(gh release list --limit 1 --json tagName,publishedAt | jq -r '.[] | "\(.tagName) \(.publishedAt)"')
                    if [ -n "$latest_release" ]; then
                        print_status "🎯 Latest Release: $latest_release" "$GREEN"
                    fi
                    break
                else
                    print_status "❌ Status: Failed ($conclusion)" "$RED"
                    echo
                    print_status "📋 Failed Jobs:" "$RED"
                    gh run view $run_id --json jobs | jq -r '.jobs[] | select(.conclusion == "failure") | "  ❌ \(.name): \(.conclusion)"'
                    break
                fi
                ;;
        esac
        
        echo
        print_status "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)" "$NC"
        sleep 10
    done
}

# Function to show deployment dashboard
show_dashboard() {
    print_status "📊 AI Platform Trainer - Deployment Dashboard" "$BLUE"
    print_status "===============================================" "$BLUE"
    echo
    
    # Recent runs
    print_status "📈 Recent Workflow Runs:" "$NC"
    gh run list --workflow="Build and Deploy Executables" --limit 5 | head -6
    echo
    
    # Latest release info
    print_status "🎯 Latest Release:" "$NC"
    gh release list --limit 1
    echo
    
    # Repository stats
    print_status "📊 Repository Stats:" "$NC"
    echo "  🌟 Stars: $(gh repo view --json stargazerCount | jq -r '.stargazerCount')"
    echo "  🍴 Forks: $(gh repo view --json forkCount | jq -r '.forkCount')"
    echo "  📥 Total Downloads: $(gh release list --json assets | jq '[.[].assets[].downloadCount] | add // 0')"
    echo
    
    # Workflow status
    print_status "⚡ Active Workflows:" "$NC"
    gh workflow list --json name,state,id | jq -r '.[] | "  \(.name): \(.state)"'
}

# Function to trigger manual deployment
trigger_deployment() {
    print_status "🚀 Triggering manual deployment..." "$YELLOW"
    gh workflow run "Build and Deploy Executables"
    print_status "✅ Deployment triggered! Use 'monitor' to track progress." "$GREEN"
}

# Function to download latest artifacts
download_artifacts() {
    print_status "📥 Downloading latest build artifacts..." "$BLUE"
    
    latest_run=$(gh run list --workflow="Build and Deploy Executables" --limit 1 --json databaseId | jq -r '.[].databaseId')
    
    if [ -z "$latest_run" ]; then
        print_status "❌ No recent runs found" "$RED"
        exit 1
    fi
    
    mkdir -p downloads
    cd downloads
    
    print_status "📦 Downloading artifacts from run $latest_run..." "$NC"
    gh run download $latest_run
    
    print_status "✅ Artifacts downloaded to ./downloads/" "$GREEN"
    ls -la
}

# Function to show help
show_help() {
    print_status "🚀 AI Platform Trainer - Deployment Tools" "$BLUE"
    echo
    print_status "Usage: $0 [command]" "$NC"
    echo
    print_status "Commands:" "$NC"
    echo "  monitor     - Monitor latest deployment workflow in real-time"
    echo "  dashboard   - Show deployment dashboard with stats"
    echo "  trigger     - Trigger manual deployment"
    echo "  download    - Download latest build artifacts"
    echo "  status      - Show current workflow status"
    echo "  logs        - Show logs from latest run"
    echo "  releases    - List recent releases"
    echo "  help        - Show this help message"
    echo
    print_status "Examples:" "$NC"
    echo "  $0 monitor    # Watch deployment progress"
    echo "  $0 dashboard  # View deployment stats"
    echo "  $0 trigger    # Start new deployment"
}

# Function to show current status
show_status() {
    run_info=$(get_latest_run)
    run_id=$(echo $run_info | cut -d' ' -f1)
    status=$(echo $run_info | cut -d' ' -f2)
    conclusion=$(echo $run_info | cut -d' ' -f3)
    
    print_status "🔍 Current Deployment Status" "$BLUE"
    print_status "Run ID: $run_id" "$NC"
    print_status "Status: $status" "$NC"
    print_status "Conclusion: ${conclusion:-running}" "$NC"
}

# Function to show logs
show_logs() {
    latest_run=$(gh run list --workflow="Build and Deploy Executables" --limit 1 --json databaseId | jq -r '.[].databaseId')
    print_status "📋 Showing logs for run $latest_run" "$BLUE"
    gh run view $latest_run --log
}

# Function to list releases
list_releases() {
    print_status "🎯 Recent Releases:" "$BLUE"
    gh release list --limit 10
}

# Main script logic
case "${1:-help}" in
    "monitor")
        monitor_workflow
        ;;
    "dashboard")
        show_dashboard
        ;;
    "trigger")
        trigger_deployment
        ;;
    "download")
        download_artifacts
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "releases")
        list_releases
        ;;
    "help"|*)
        show_help
        ;;
esac