import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const cardVariants = cva(
  "flex flex-col gap-6 py-6",
  {
    variants: {
      variant: {
        default: "bg-card text-card-foreground rounded-xl border shadow-sm",
        // 2005 Retro Gaming Variants
        retro: "bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4] text-[#5D4037] border-4 border-[#5D4037] rounded-2xl shadow-[6px_6px_0_#3E2723,0_8px_16px_rgba(0,0,0,0.15)] relative",
        "retro-yellow": "bg-gradient-to-b from-[#FFF9C4] to-[#FFECB3] text-[#5D4037] border-4 border-[#FFA000] rounded-2xl shadow-[5px_5px_0_#FF8F00,0_6px_12px_rgba(0,0,0,0.15)]",
        "retro-pink": "bg-gradient-to-b from-[#FCE4EC] to-[#F8BBD9] text-[#880E4F] border-4 border-[#C2185B] rounded-2xl shadow-[5px_5px_0_#880E4F,0_6px_12px_rgba(0,0,0,0.15)]",
        "retro-green": "bg-gradient-to-b from-[#E8F5E9] to-[#C8E6C9] text-[#1B5E20] border-4 border-[#388E3C] rounded-2xl shadow-[5px_5px_0_#1B5E20,0_6px_12px_rgba(0,0,0,0.15)]",
        "retro-blue": "bg-gradient-to-b from-[#E3F2FD] to-[#BBDEFB] text-[#0D47A1] border-4 border-[#1976D2] rounded-2xl shadow-[5px_5px_0_#0D47A1,0_6px_12px_rgba(0,0,0,0.15)]",
        "retro-purple": "bg-gradient-to-b from-[#F3E5F5] to-[#E1BEE7] text-[#4A148C] border-4 border-[#7B1FA2] rounded-2xl shadow-[5px_5px_0_#4A148C,0_6px_12px_rgba(0,0,0,0.15)]",
        "retro-hero": "bg-gradient-to-br from-[#9B59B6] via-[#E91E63] to-[#FF5722] text-white border-4 border-[#4A148C] rounded-2xl shadow-[6px_6px_0_#311B92,0_8px_16px_rgba(0,0,0,0.25)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Card({ 
  className, 
  variant, 
  ...props 
}: React.ComponentProps<"div"> & VariantProps<typeof cardVariants>) {
  return (
    <div
      data-slot="card"
      className={cn(cardVariants({ variant, className }))}
      {...props}
    />
  )
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2 px-6 has-data-[slot=card-action]:grid-cols-[1fr_auto] [.border-b]:pb-6",
        className
      )}
      {...props}
    />
  )
}

const cardTitleVariants = cva(
  "leading-none font-semibold",
  {
    variants: {
      variant: {
        default: "",
        retro: "font-display text-lg tracking-wide",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function CardTitle({ 
  className, 
  variant, 
  ...props 
}: React.ComponentProps<"div"> & VariantProps<typeof cardTitleVariants>) {
  return (
    <div
      data-slot="card-title"
      className={cn(cardTitleVariants({ variant, className }))}
      {...props}
    />
  )
}

function CardDescription({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-description"
      className={cn("text-muted-foreground text-sm", className)}
      {...props}
    />
  )
}

function CardAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-action"
      className={cn(
        "col-start-2 row-span-2 row-start-1 self-start justify-self-end",
        className
      )}
      {...props}
    />
  )
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-content"
      className={cn("px-6", className)}
      {...props}
    />
  )
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn("flex items-center px-6 [.border-t]:pt-6", className)}
      {...props}
    />
  )
}

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardAction,
  CardDescription,
  CardContent,
  cardVariants,
  cardTitleVariants,
}
