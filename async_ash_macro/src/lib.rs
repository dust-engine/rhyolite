extern crate proc_macro;
use std::{cell::RefCell, borrow::Borrow};

use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    token, ExprUnsafe,
};

struct ExprGpuAsync {
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            stmts: syn::Block::parse_within(input)?,
        })
    }
}


struct TransformContext<'a, A: FnMut(&syn::ExprAwait) -> syn::Expr, M: FnMut(&syn::ExprMacro) -> syn::Expr, R: FnMut(&syn::ExprReturn) -> Option<syn::Expr>> {
    async_transform: &'a mut A,
    macro_transform: &'a mut M,
    return_transform: &'a mut R,
}

fn transform_stmt_asyncs(
    stmt: &syn::Stmt,
    transform_context: &mut TransformContext<'_, impl FnMut(&syn::ExprAwait) -> syn::Expr, impl FnMut(&syn::ExprMacro) -> syn::Expr, impl FnMut(&syn::ExprReturn) -> Option<syn::Expr>>
) -> syn::Stmt {
    use syn::{Block, Expr, Pat, Stmt};
    fn transform_block(
        block: &Block,
        transform_context: &mut TransformContext<'_, impl FnMut(&syn::ExprAwait) -> syn::Expr, impl FnMut(&syn::ExprMacro) -> syn::Expr, impl FnMut(&syn::ExprReturn) -> Option<syn::Expr>>
    ) -> Block {
        Block {
            brace_token: block.brace_token.clone(),
            stmts: block
                .stmts
                .iter()
                .map(move |stmt| transform_stmt(stmt, transform_context))
                .collect(),
        }
    }
    fn transform_pattern(
        pat: &Pat,
        transform_context: &mut TransformContext<'_, impl FnMut(&syn::ExprAwait) -> syn::Expr, impl FnMut(&syn::ExprMacro) -> syn::Expr, impl FnMut(&syn::ExprReturn) -> Option<syn::Expr>>
    ) -> Pat {
        match pat {
            Pat::Box(pat) => Pat::Box(syn::PatBox {
                pat: Box::new(transform_pattern(&pat.pat, transform_context)),
                ..pat.clone()
            }),
            Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                subpat: ident.subpat.as_ref().map(|(at, subpat)| {
                    (
                        at.clone(),
                        Box::new(transform_pattern(subpat, transform_context)),
                    )
                }),
                ..ident.clone()
            }),
            Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                expr: Box::new(transform_expr(&lit.expr, transform_context)),
                ..lit.clone()
            }),
            Pat::Or(clause) => Pat::Or(syn::PatOr {
                cases: clause
                    .cases
                    .iter()
                    .map(|pat| transform_pattern(pat, transform_context))
                    .collect(),
                ..clause.clone()
            }),
            Pat::Range(range) => Pat::Range(syn::PatRange {
                lo: Box::new(transform_expr(&range.lo, transform_context)),
                hi: Box::new(transform_expr(&range.hi, transform_context)),
                ..range.clone()
            }),
            Pat::Reference(r) => Pat::Reference(syn::PatReference {
                pat: Box::new(transform_pattern(&r.pat, transform_context)),
                ..r.clone()
            }),
            Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                elems: slice
                    .elems
                    .iter()
                    .map(|pat| transform_pattern(pat, transform_context))
                    .collect(),
                ..slice.clone()
            }),
            Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldPat {
                        pat: Box::new(transform_pattern(&f.pat, transform_context)),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|pat| transform_pattern(pat, transform_context))
                    .collect(),
                ..tuple.clone()
            }),
            Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                pat: syn::PatTuple {
                    elems: tuple
                        .pat
                        .elems
                        .iter()
                        .map(|pat| transform_pattern(pat, transform_context))
                        .collect(),
                    ..tuple.pat.clone()
                },
                ..tuple.clone()
            }),
            Pat::Type(ty) => Pat::Type(syn::PatType {
                pat: Box::new(transform_pattern(&ty.pat, transform_context)),
                ..ty.clone()
            }),
            _ => pat.clone(),
        }
    }
    fn transform_expr(
        expr: &Expr,
        transform_context: &mut TransformContext<'_, impl FnMut(&syn::ExprAwait) -> syn::Expr, impl FnMut(&syn::ExprMacro) -> syn::Expr, impl FnMut(&syn::ExprReturn) -> Option<syn::Expr>>
    ) -> Expr {
        match expr {
            Expr::Array(arr) => Expr::Array(syn::ExprArray {
                elems: arr
                    .elems
                    .iter()
                    .map(|expr| transform_expr(expr, transform_context))
                    .collect(),
                ..arr.clone()
            }),
            Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                left: Box::new(transform_expr(&*assign.left, transform_context)),
                right: Box::new(transform_expr(&*assign.right, transform_context)),
                ..assign.clone()
            }),
            Expr::AssignOp(assign) => Expr::AssignOp(syn::ExprAssignOp {
                left: Box::new(transform_expr(&*assign.left, transform_context)),
                right: Box::new(transform_expr(&*assign.right, transform_context)),
                ..assign.clone()
            }),
            Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                left: Box::new(transform_expr(&*binary.left, transform_context)),
                right: Box::new(transform_expr(&*binary.right, transform_context)),
                ..binary.clone()
            }),
            Expr::Block(block) => Expr::Block(syn::ExprBlock {
                block: transform_block(&block.block, transform_context),
                ..block.clone()
            }),
            Expr::Call(call) => Expr::Call(syn::ExprCall {
                func: Box::new(transform_expr(&*call.func, transform_context)),
                args: call
                    .args
                    .iter()
                    .map(|expr| transform_expr(expr, transform_context))
                    .collect(),
                ..call.clone()
            }),
            Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                expr: Box::new(transform_expr(&*cast.expr, transform_context)),
                ..cast.clone()
            }),
            Expr::Field(field) => Expr::Field(syn::ExprField {
                base: Box::new(transform_expr(&*field.base, transform_context)),
                ..field.clone()
            }),
            Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                pat: transform_pattern(&l.pat, transform_context),
                expr: Box::new(transform_expr(&*l.expr, transform_context)),
                body: transform_block(&l.body, transform_context),
                ..l.clone()
            }),
            Expr::Group(group) => Expr::Group(syn::ExprGroup {
                expr: Box::new(transform_expr(&group.expr, transform_context)),
                ..group.clone()
            }),
            Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                cond: Box::new(transform_expr(&if_stmt.cond, transform_context)),
                then_branch: transform_block(&if_stmt.then_branch, transform_context),
                else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                    (
                        else_token.clone(),
                        Box::new(transform_expr(&expr, transform_context)),
                    )
                }),
                ..if_stmt.clone()
            }),
            Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                expr: Box::new(transform_expr(&*index_expr.expr, transform_context)),
                index: Box::new(transform_expr(&*index_expr.index, transform_context)),
                ..index_expr.clone()
            }),
            Expr::Let(l) => Expr::Let(syn::ExprLet {
                pat: transform_pattern(&l.pat, transform_context),
                expr: Box::new(transform_expr(&*l.expr, transform_context)),
                ..l.clone()
            }),
            Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                body: transform_block(&loop_stmt.body, transform_context),
                ..loop_stmt.clone()
            }),
            Expr::Macro(m) => (transform_context.macro_transform)(m),
            Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                expr: Box::new(transform_expr(&*match_expr.expr, transform_context)),
                arms: match_expr
                    .arms
                    .iter()
                    .map(|arm| syn::Arm {
                        pat: transform_pattern(&arm.pat, transform_context),
                        guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                            (
                                guard_token.clone(),
                                Box::new(transform_expr(&expr, transform_context)),
                            )
                        }),
                        body: Box::new(transform_expr(&arm.body, transform_context)),
                        ..arm.clone()
                    })
                    .collect(),
                ..match_expr.clone()
            }),
            Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                receiver: Box::new(transform_expr(&call.receiver, transform_context)),
                args: call
                    .args
                    .iter()
                    .map(|expr| transform_expr(expr, transform_context))
                    .collect(),
                ..call.clone()
            }),
            Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                expr: Box::new(transform_expr(&paren.expr, transform_context)),
                ..paren.clone()
            }),
            Expr::Range(range) => Expr::Range(syn::ExprRange {
                from: range
                    .from
                    .as_ref()
                    .map(|f| Box::new(transform_expr(&f, transform_context))),
                to: range
                    .to
                    .as_ref()
                    .map(|t| Box::new(transform_expr(&t, transform_context))),
                ..range.clone()
            }),
            Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                expr: Box::new(transform_expr(&reference.expr, transform_context)),
                ..reference.clone()
            }),
            Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                expr: Box::new(transform_expr(&repeat.expr, transform_context)),
                len: Box::new(transform_expr(&repeat.len, transform_context)),
                ..repeat.clone()
            }),
            Expr::Return(ret) => {
                if let Some(expr) = (transform_context.return_transform)(ret) {
                    expr
                } else {
                    Expr::Return(syn::ExprReturn {
                        expr: ret
                            .expr
                            .as_ref()
                            .map(|e| Box::new(transform_expr(&e, transform_context))),
                        ..ret.clone()
                    })
                }
            },
            Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldValue {
                        member: f.member.clone(),
                        expr: transform_expr(&f.expr, transform_context),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Expr::Try(s) => Expr::Try(syn::ExprTry {
                expr: Box::new(transform_expr(&s.expr, transform_context)),
                ..s.clone()
            }),
            Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|expr| transform_expr(expr, transform_context))
                    .collect(),
                ..tuple.clone()
            }),
            Expr::Type(type_expr) => Expr::Type(syn::ExprType {
                expr: Box::new(transform_expr(&type_expr.expr, transform_context)),
                ..type_expr.clone()
            }),
            Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                expr: Box::new(transform_expr(&unary.expr, transform_context)),
                ..unary.clone()
            }),
            Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(ExprUnsafe {
                block: transform_block(&unsafe_stmt.block, transform_context),
                ..unsafe_stmt.clone()
            }),
            Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                cond: Box::new(transform_expr(&while_stmt.cond, transform_context)),
                body: transform_block(&while_stmt.body, transform_context),
                ..while_stmt.clone()
            }),
            Expr::Await(await_expr) => (transform_context.async_transform)(await_expr),
            _ => expr.clone(),
        }
    }
    fn transform_stmt(
        stmt: &Stmt,
        transform_context: &mut TransformContext<'_, impl FnMut(&syn::ExprAwait) -> syn::Expr, impl FnMut(&syn::ExprMacro) -> syn::Expr, impl FnMut(&syn::ExprReturn) -> Option<syn::Expr>>
    ) -> Stmt {
        match stmt {
            Stmt::Local(local) => Stmt::Local(syn::Local {
                pat: transform_pattern(&local.pat, transform_context),
                init: local.init.as_ref().map(|(eq_token, expr)| {
                    (
                        eq_token.clone(),
                        Box::new(transform_expr(&expr, transform_context)),
                    )
                }),
                ..local.clone()
            }),
            Stmt::Expr(expr) => {
                Stmt::Expr(transform_expr(&expr, transform_context))
            },
            Stmt::Semi(expr, semi) => {
                Stmt::Semi(transform_expr(expr, transform_context), semi.clone())
            }
            _ => stmt.clone(),
        }
    }
    transform_stmt(stmt, transform_context)
}

#[proc_macro]
pub fn commands(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro_commands(input.into()).into()
}

fn proc_macro_commands(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };

    let mut current_import_id: usize = 0;
    let mut import_bindings = proc_macro2::TokenStream::new();
    let mut import_drops = RefCell::new(proc_macro2::TokenStream::new());
    let mut proc_macro_import_expr = |input_tokens: &proc_macro2::TokenStream| -> proc_macro2::TokenStream {
        let global_res_variable_name = quote::format_ident!("__future_res_{}", current_import_id);
        current_import_id += 1;
        let output_tokens = quote::quote! {
            ::async_ash::future::Res::new(0, &#global_res_variable_name)
        };
        import_bindings.extend(quote::quote! {
            let #global_res_variable_name = #input_tokens;
        });
        import_drops.borrow_mut().extend(quote::quote! {
            drop(#global_res_variable_name);
        });
        output_tokens
    };

    // Count the total number of awaits in the input

    let mut current_await_index: usize = 0;
    let mut awaited_future_bindings = proc_macro2::TokenStream::new();
    let awaited_future_drops = RefCell::new(proc_macro2::TokenStream::new());
    let mut inner_closure_stmts: Vec<_> = input
    .stmts
    .iter()
    .map(|stmt| {
        transform_stmt_asyncs(stmt, &mut TransformContext {
            async_transform: &mut |a| {
                let global_future_variable_name = quote::format_ident!("__future_{}", current_await_index);
                current_await_index += 1;

                awaited_future_bindings.extend(quote::quote! {
                    let mut #global_future_variable_name;
                });
                awaited_future_drops.borrow_mut().extend(quote::quote! {
                    drop(#global_future_variable_name);
                });

                let base = a.base.clone();


                // For each future, we first call init on it. This is a no-op for most futures, but for
                // (nested) block futures, this is going to move the future to its first yield point.
                // At that point it should yield the context of the first future.
                let tokens = quote::quote! {
                    {
                        #global_future_variable_name = #base;
                        let mut fut_pinned = unsafe { std::pin::Pin::new_unchecked(&mut #global_future_variable_name) };
                        fut_pinned.as_mut().init();
                        let mut ctx = Default::default();
                        fut_pinned.as_mut().context(&mut ctx);
                        yield std::ops::GeneratorState::Yielded(ctx);
                        loop {
                            match fut_pinned.as_mut().record(__fut_ctx) {
                                std::task::Poll::Ready(v) => break v,
                                std::task::Poll::Pending => {
                                    let mut ctx = Default::default();
                                    fut_pinned.as_mut().context(&mut ctx);
                                    yield std::ops::GeneratorState::Yielded(ctx);
                                },
                            };
                        }
                    }
                };
                syn::parse2(tokens).unwrap()
            },
            macro_transform: &mut |mac| {
                let path = &mac.mac.path;
                if path.segments.len() != 1 {
                    return syn::Expr::Macro(mac.clone());
                }
                match path.segments[0].ident.to_string().as_str() {
                    "import" => syn::Expr::Verbatim(proc_macro_import_expr(&mac.mac.tokens)),
                    _ => syn::Expr::Macro(mac.clone())
                }
            },
            return_transform: &mut |ret| {
                // Transform each return statement into a yield, drops, and return.
                // We use RefCell on awaited_future_drops and import_drops so that they can be read while being modified.
                // This ensures that we won't drop uninitialized values.'
                // The executor will stop the execution as soon as it reaches the first `yield Complete` statement.
                // Drops are written between the yield and return, so these values are retained inside the generator
                // until the generator itself was dropped.
                // We drop the generator only after semaphore was signaled from within the queue.
                let returned_item = ret.expr.as_ref();
                let awaited_future_drops = &*awaited_future_drops.borrow();
                let import_drops = &*import_drops.borrow();
                let token_stream = quote::quote!(
                    {
                        yield std::ops::GeneratorState::Complete(#returned_item);
                        #awaited_future_drops
                        #import_drops
                        return;
                    }
                );
                let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
                Some(syn::Expr::Block(block))
            }
        })
    })
    .collect();

    if let Some(last) = inner_closure_stmts.last_mut() {
        
        let awaited_future_drops = &*awaited_future_drops.borrow();
        let import_drops = &*import_drops.borrow();
        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!(
                {
                    yield std::ops::GeneratorState::Complete(#expr);
                    #awaited_future_drops
                    #import_drops
                    return;
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            *expr = syn::Expr::Block(block);
        } else {
            let token_stream = quote::quote!(
                {
                    yield std::ops::GeneratorState::Complete(());
                    #awaited_future_drops
                    #import_drops
                    return;
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Block(block)))
        }
    }

    quote::quote! {
        async_ash::future::GPUCommandBlock::new(static |__fut_ctx| {
            #import_bindings
            #awaited_future_bindings
            #(#inner_closure_stmts)*
        })
    }
}

struct MacroJoin {
    pub exprs: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}
impl Parse for MacroJoin {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroJoin {
            exprs: syn::punctuated::Punctuated::parse_separated_nonempty(input)?,
        })
    }
}

#[proc_macro]
pub fn join(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro_join(input.into()).into()
}

fn proc_macro_join(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<MacroJoin>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    if input.exprs.len() == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "Expects at least one argument",
        )
        .to_compile_error();
    }
    if input.exprs.len() == 1 {
        return input.exprs[0].clone().into_token_stream();
    }

    let mut token_stream = proc_macro2::TokenStream::new();
    token_stream.extend(input.exprs[0].clone().into_token_stream());

    // a.join(b).join(c)...
    for item in input.exprs.iter().skip(1) {
        token_stream.extend(quote::quote! {.join(#item)}.into_iter());
    }

    let num_expressions = input.exprs.len();

    // __join_0, __join_1, __join_2, ...
    let output_expression = (0..num_expressions).map(|i| quote::format_ident!("__join_{}", i));

    let input_expression = {
        use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
        let mut t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
            let mut t = TokenStream::new();
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_0",
                Span::call_site(),
            ))));
            t.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_1",
                Span::call_site(),
            ))));
            t
        })));
        (2..num_expressions).for_each(|i| {
            let prev = t.take().unwrap().into_token_stream();
            t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
                let mut a = TokenStream::new();
                a.extend(Some(prev));
                a.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
                a.extend(Some(TokenTree::Ident(quote::format_ident!("__join_{}", i))));
                a
            })));
        });
        t
    };
    token_stream
        .extend(quote::quote! {.map(|#input_expression| (#(#output_expression),*))}.into_iter());
    token_stream
}
